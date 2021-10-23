import argparse
import logging
import torch
import random
import torch.nn as nn
from torch.nn.functional import normalize
from torch.nn.utils.rnn import pad_sequence

import transformers
import datasets

import data
#import eval_util

from dp_pred import dp_single, backtrack

class TagWordModel(nn.Module):
    def __init__(self, opt):
        super(TagWordModel, self).__init__()
        self.register_buffer('dummy', torch.Tensor(1, 1).fill_(-float("inf")))
        self.bert = transformers.BertModel.from_pretrained(opt.bert_model, cache_dir=CACHEDIR)
        for lay in self.bert.encoder.layer:
            lay.output.dropout.p = args.drop
        self.dropout = nn.Dropout(args.drop)

    def get_word_reps(self, x, C, shard_bsz=None):
        """
        x - bsz x max_wpcs
        C - bsz x T x num_pcs, binary
        if shard_bsz is not None, we assume we can detach things
        returns bsz x T x dim
        """
        if shard_bsz is not None:
            wreps = []
            xsplits = torch.split(x, shard_bsz, dim=0)
            Csplits = torch.split(C, shard_bsz, dim=0)
            for i, xsplit in enumerate(xsplits):
                # get shardsz x npcs x dim bertrep
                mask = (xsplit != 0).long()
                bertrep = self.bert(input_ids=xsplit, attention_mask=mask)["last_hidden_state"]
                splitrep = torch.bmm(Csplits[i], bertrep) # shardsz x T x dim
                wreps.append(splitrep) # shardsz x T x dim
            wreps = torch.cat(wreps, 0) # bsz x T x dim
        else:
            mask = (x != 0).long()
            bertrep = self.bert( # bsz x num_pcs x dim
                input_ids=x, attention_mask=mask)["last_hidden_state"]
            # get word reps by selecting or adding pieces
            wreps = torch.bmm(C, bertrep) # bsz x T x dim
        return wreps


def lse_loss(scores, targets, dummy):
    """
    scores - bsz x nchoices, log-softmaxed
    targets - bsz x max_correct
    dummy - 1 x 1 dummy val (usually -inf)
    """
    # concatenate on dummy
    scores = torch.cat([scores, dummy.expand(scores.size(0), -1)], 1) # bsz x nchoices+1
    crct_scores = scores.gather(1, targets) # bsz x max_correct
    loss = torch.logsumexp(crct_scores, dim=1) # bsz
    return -loss.sum()

def get_batch_loss(batch_reps, ne_reps, tgts, dummy, args, val=False):
    """
    batch_reps - bsz x T x dim
    ne_reps - nesz*max_len x dim
    tgts - bsz x T x max_correct
    """
    bsz, T, hidsize = batch_reps.size()

    if args.cosine:
        ne_reps = normalize(ne_reps, p=2, dim=1) # nesz*max_len x dim
        batch_reps = normalize(batch_reps, p=2, dim=2)

    scores = torch.log_softmax( # bsz*T x nesz*max_len
        torch.mm(batch_reps.view(-1, hidsize), ne_reps.t()), dim=1)
    loss = lse_loss(scores, tgts.view(scores.size(0), -1), dummy)
    lossitem = loss.item()
    if not val:
        loss.div(bsz*T).backward()
    return lossitem

def get_batch_preds(batch_reps, ne_reps, tag2mask, args, ne_tag_seqs=None):
    """
    batch_reps - bsz x T x dim
    ne_reps - nesz*max_len x dim
    tag2mask - list of (tagtype, mask) tuples
    returns bsz x T tag predictions
    """
    bsz, T, hidsize = batch_reps.size()
    if args.cosine:
        ne_reps = normalize(ne_reps, p=2, dim=1) # nesz*max_len x dim
        batch_reps = normalize(batch_reps, p=2, dim=2)

    scores = torch.log_softmax( # bsz*T x nesz*max_len
        torch.mm(batch_reps.view(-1, hidsize), ne_reps.t()), dim=1)
    # sum over all neighbor tokens w/ the same tag
    tagscores = []
    for tagtype, mask in tag2mask:
        tagscores.append(torch.logsumexp(scores + mask, dim=1)) # bsz*T
    num_copies = None
    if args.just_eval and args.dp_pred:
        assert bsz == 1
        labe_probs = torch.stack(tagscores) # num_tags x bsz*T
        labe_probs = labe_probs.t().exp()
        labe2idx = {tagtype: i for i, (tagtype, _) in enumerate(tag2mask)}
        idx2tag = {i: tagtype for i, (tagtype, _) in enumerate(tag2mask)}
        _, bps, _, keysbylen, _ = dp_single(labe_probs, ne_tag_seqs, labe2idx, args.c)
        pred, num_copies = backtrack(bps, keysbylen, idx2tag)
        preds = [pred]
    else:
        # get a single tag pred for each token
        _, preds = torch.stack(tagscores).max(0) # bsz*T
        preds = preds.view(bsz, T)
        # map back to tags (and transpose)
        idx2tag = {i: tagtype for i, (tagtype, _) in enumerate(tag2mask)}
        preds = [[idx2tag[idx.item()] for idx in row] for row in preds]
    return preds, num_copies

def train(sentdb, model, optim, scheduler, device, args):
    model.train()
    total_loss, total_preds = 0.0, 0
    perm = torch.randperm(len(sentdb.minibatches))
    for i, idx in enumerate(perm):
        optim.zero_grad()
        x, neighbs, Cx, Cn, tgts = sentdb.word_batch(idx.item())
        x, neighbs = x.to(device), neighbs.to(device)
        Cx, Cn, tgts = Cx.to(device), Cn.to(device), tgts.to(device)
        bsz, T = x.size()
        # Tne, nesz = neighbs.size()
        batch_reps = model.get_word_reps(x, Cx) # bsz x T  x dim
        if args.detach_db:
            with torch.no_grad():
                ne_reps = model.get_word_reps(  # nesz x max_len x dim
                    neighbs, Cn, shard_bsz=args.pred_shard_size)
        else:
            ne_reps = model.get_word_reps(neighbs, Cn) # max_len x nesz x dim
        if args.drop_db:
            ne_reps = model.dropout(ne_reps)
        total_loss += get_batch_loss(batch_reps, ne_reps.view(-1, ne_reps.size(2)),
                                     tgts, model.dummy, args)
        optim.step()
        scheduler.step()
        total_preds += (T*bsz)
        if (i+1) % args.log_interval == 0:
            print("batch", i+1, "loss:", total_loss/total_preds)
    return total_loss/total_preds


def do_fscore(sentdb, model, metric, device, args):
    """
    micro-avgd segment-level f1-score
    """
    model.eval()
    #total_pred, total_gold, total_crct = 0.0, 0.0, 0.0
    for i in range(len(sentdb.val_minibatches)):
        x, neighbs, Cx, Cn, tag2mask, gold = sentdb.predword_batch(
            i, args.eval_ne_per_sent)
        x, neighbs = x.to(device), neighbs.to(device)
        Cx, Cn, = Cx.to(device), Cn.to(device)
        tag2mask = [(tag, mask.to(device)) for (tag, mask) in tag2mask]
        # Tne, nesz = neighbs.size()
        batch_reps = model.get_word_reps(x, Cx) # bsz x T x dim
        ne_reps = model.get_word_reps(  # nesz x max_len x dim
            neighbs, Cn, shard_bsz=args.pred_shard_size)
        preds = get_batch_preds(batch_reps, ne_reps.view(-1, ne_reps.size(2)),
                                tag2mask, args)[0] # bsz x T
        goldlabels = [[sentdb.label_list[glabel] for glabel in goldseq]
                      for goldseq in gold]
        predlabels = [[sentdb.label_list[plabel] for plabel in predseq]
                      for predseq in preds]
        metric.add_batch(predictions=predlabels, references=goldlabels)
    results = metric.compute() # should delete cached predictions/golds
    return {"precision": results["overall_precision"],
            "recall": results["overall_recall"],
            "f1": results["overall_f1"],
            "accuracy": results["overall_accuracy"]}


# the point of this is so we don't mix neighbors based on other stuff in the minibatch
def do_single_fscore(sentdb, model, device, args):
    """
    micro-avgd segment-level f1-score
    """
    model.eval()
    metric = datasets.load_metric("seqeval")
    print("predicting on", len(sentdb.vsent_words), "sentences")
    for i in range(len(sentdb.vsent_words)):
        #if i > 20:
        #    break
        if i % 200 == 0:
            print("sent", i)
        x, neighbs, Cx, Cn, tag2mask, gold, ne_tags = sentdb.pred_single_batch(
            i, args.eval_ne_per_sent)
        x, neighbs = x.to(device), neighbs.to(device)
        Cx, Cn, = Cx.to(device), Cn.to(device)
        tag2mask = [(tag, mask.to(device)) for (tag, mask) in tag2mask]
        batch_reps = model.get_word_reps(x, Cx) # 1 x T x dim
        ne_reps = model.get_word_reps(  # nesz x max_len x dim
            neighbs, Cn, shard_bsz=args.pred_shard_size)
        preds, ncopies = get_batch_preds(batch_reps, ne_reps.view(-1, ne_reps.size(2)),
                                         tag2mask, args, ne_tag_seqs=ne_tags) # 1 x T
        goldlabels = [[sentdb.label_list[glabel] for glabel in goldseq]
                      for goldseq in gold]
        predlabels = [[sentdb.label_list[plabel] for plabel in predseq]
                      for predseq in preds]
        metric.add_batch(predictions=predlabels, references=goldlabels)
    results = metric.compute()
    print(results)


parser = argparse.ArgumentParser()
parser.add_argument("-bert_model", default="bert-base-cased", type=str,
                    choices=["bert-base-uncased", "bert-base-cased"])
parser.add_argument("-tagkey", default="ner_tags", type=str,
                    choices=["ner_tags", "pos_tags", "chunk_tags"])
parser.add_argument("-db_fi", default=None, type=str, help="")
parser.add_argument("-load_saved_db", action='store_true', help="")
parser.add_argument('-bsz', type=int, default=16, help='')
parser.add_argument('-seed', type=int, default=3435, help='')
parser.add_argument("-cuda", action='store_true', help="")
parser.add_argument("-train_from", default="", type=str, help="")
parser.add_argument("-nosplit_parenth", action='store_true', help="")

parser.add_argument("-lr", default=5e-5, type=float, help="The initial learning rate for Adam.")
parser.add_argument("-drop", default=0.4, type=float, help="")
parser.add_argument("-clip", default=1, type=float, help="")
parser.add_argument("-wd", default=0.0001, type=float, help="")
parser.add_argument("-warmup_prop", default=0.2, type=float)
parser.add_argument('-epochs', type=int, default=10, help='')
parser.add_argument('-log_interval', type=int, default=100, help='')
parser.add_argument('-save', type=str, default='', help='path to save the final model')

parser.add_argument('-ne_per_sent', type=int, default=50, help='')
parser.add_argument("-random_tr_ne", action='store_true', help="")
parser.add_argument('-eval_ne_per_sent', type=int, default=100, help='')
parser.add_argument("-acc_eval", action='store_true', help="")
parser.add_argument('-pred_shard_size', type=int, default=64, help='')
parser.add_argument("-cosine", action='store_true', help="")
parser.add_argument("-align_strat", type=str, default="first",
                    choices=["sum", "first", "last"])
parser.add_argument("-drop_db", action='store_true', help="")
parser.add_argument("-detach_db", action='store_true', help="")
parser.add_argument("-subsample_all", action='store_true', help="")
parser.add_argument("-cachedir", type=str, default="/var/tmp/local/users/sjw68/hfstuff")
parser.add_argument("-just_eval", type=str, default=None,
                    choices=["dev", "dev-newne"], help="")
parser.add_argument("-transfer", action='store_true', help="")

parser.add_argument("-dp_pred", action='store_true', help="")
parser.add_argument("-c", default=1, type=float, help="")

if __name__ == "__main__":
    args = parser.parse_args()
    print(args)
    CACHEDIR = None if args.cachedir.lower() == "none" else args.cachedir

    logger = logging.getLogger(__name__)
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO)
    logger.setLevel(logging.INFO)

    torch.set_num_threads(2)
    #torch.manual_seed(args.seed)
    transformers.set_seed(args.seed)
    random.seed(args.seed)

    if args.transfer:
        assert not args.load_saved_db
        assert args.just_eval is not None

    if torch.cuda.is_available():
        if not args.cuda:
            print("WARNING: You have a CUDA device, so you should probably run with --cuda")
    device = torch.device("cuda" if args.cuda else "cpu")
    #device = accelerator.device

    if len(args.train_from) > 0:
        print("loading model from", args.train_from)
        saved_stuff = torch.load(args.train_from)
        saved_args = saved_stuff["opt"]
        model = TagWordModel(args)
        model.load_state_dict(saved_stuff["state_dict"])
    else:
        model = TagWordModel(args)

    model = model.to(device)

    nosplits = ("[UNK]", "[SEP]", "[PAD]", "[CLS]", "[MASK]")
    if args.nosplit_parenth:
        nosplits = nosplits + ("-LPR-", "-RPR-")
    tokenizer = transformers.BertTokenizerFast.from_pretrained(
        args.bert_model, cache_dir=CACHEDIR)

    if args.load_saved_db and args.db_fi is not None:
        logger.info("loading db from {}".format(args.db_fi))
        sentdb = data.SentDB(args.tagkey, tokenizer, CACHEDIR, path=args.db_fi,
                             align_strat=args.align_strat, subsample_all=args.subsample_all)
    else:
        logger.info("making sentdb...")
        sentdb = data.SentDB(args.tagkey, tokenizer, CACHEDIR, align_strat=args.align_strat,
                             subsample_all=args.subsample_all)
        logger.info("made sentdb...")

        nebert = model.bert
        if args.transfer and "newne" not in args.just_eval:
            nebert = transformers.BertModel.from_pretrained(args.bert_model, cache_dir=CACHEDIR)
            nebert = nebert.to(device)

        def avg_bert_emb(x):
            mask = (x != 0)
            rep = nebert(input_ids=x, attention_mask=mask.long())["last_hidden_state"] # bsz x T x d
            mask = mask.float().unsqueeze(2) # bsz x T x 1
            avgs = (rep * mask).sum(1) / mask.sum(1) # bsz x hid
            return avgs

        nebsz, nne = 2048, 200
        # we always compute neighbors w/ cosine; seems to be a bit better
        model.eval()
        logger.info("computing top neighbs...")
        sentdb.compute_top_neighbs(nebsz, avg_bert_emb, nne, device, cosine=True,
                                   ignore_trne=args.transfer)
        logger.info("computed top neighbs...")
        if not args.transfer:
            #import ipdb
            #ipdb.set_trace()
            logger.info("making train minibatches...")
            sentdb.make_minibatches(args.bsz, args.ne_per_sent, random_tr_ne=args.random_tr_ne)
            logger.info("made {:d} train minibatches...".format(len(sentdb.minibatches)))
            logger.info("making val minibatches...")
            sentdb.make_minibatches(args.bsz, args.ne_per_sent, val=True)
            logger.info("made {:d} val minibatches...".format(len(sentdb.val_minibatches)))
            if args.db_fi is not None:
                logger.info("saving db to {}".format(args.db_fi))
                sentdb.save(args.db_fi)
            model.train()

    if args.just_eval is not None:
        import sys

        if not args.transfer and args.just_eval != "dev": # if transfer we already did this
            # need to recompute new neighbors
            if "newne" in args.just_eval:
                bert = model.bert
            else:
                bert = transformers.BertModel.from_pretrained(args.bert_model, cache_dir=CACHEDIR)
                bert = bert.to(device)
            bert.eval()

            def newne_avg_bert_emb(x):
                mask = (x != 0)
                #rep, _ = bert(x, attention_mask=mask.long(), output_all_encoded_layers=False)
                rep = bert(input_ids=x, attention_mask=mask.long())["last_hidden_state"]
                mask = mask.float().unsqueeze(2) # bsz x T x 1
                avgs = (rep * mask).sum(1) / mask.sum(1) # bsz x hid
                return avgs

            # note that ne_bsz and nne are just used for recomputing neighbors
            # and for the max ne stored per sent, resp. we control how many ne
            # are used at prediction time w/ eval_ne_per_sent
            #sentdb.replace_val_w_test(args.val_sent_fi, args.val_tag_fi, tokenizer,
            sentdb.replace_val_w_test(None, None, tokenizer,
                                      newne_avg_bert_emb, device,
                                      ne_bsz=1024, nne=200)
        with torch.no_grad():
            prec, rec, f1 = do_single_fscore(sentdb, model, device, args)
            print("Eval: | P: {:3.5f} / R: {:3.5f} / F: {:3.5f}".format(
                prec, rec, f1))
            sys.exit(0)



    # Prepare optimizer
    param_optimizer = list(model.named_parameters())

    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer
                    if not any(nd in n for nd in no_decay)], 'weight_decay': args.wd},
        {'params': [p for n, p in param_optimizer
                    if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}]

    num_train_steps = args.epochs*len(sentdb.minibatches)
    logger.info(f" total train steps = {num_train_steps}")
    # there's a ton of other options like grad clipping, momentum stuff, schedules, etc
    optimizer = transformers.AdamW(optimizer_grouped_parameters, lr=args.lr)

    num_warmup_steps = int(num_train_steps*args.warmup_prop)
    lr_scheduler = transformers.get_scheduler(
        "linear", optimizer=optimizer, num_warmup_steps=num_warmup_steps,
        num_training_steps=num_train_steps)

    metric = datasets.load_metric("seqeval")

    bestloss = float("inf")
    for ep in range(args.epochs):
        trloss = train(sentdb, model, optimizer, lr_scheduler, device, args)
        logger.info("Epoch {:3d} | train loss {:8.3f}".format(ep, trloss))
        with torch.no_grad():
            res = do_fscore(sentdb, model, metric, device, args)
            logger.info("Epoch {:3d} | P: {:3.5f} / R: {:3.5f} / F: {:3.5f}".format(
                ep, res["precision"], res["recall"], res["f1"]))
            valloss = -res["f1"]
        if valloss < bestloss:
            bestloss = valloss
            if len(args.save) > 0:
                logger.info("saving to {}".format(args.save))
                torch.save({"opt": args, "state_dict": model.state_dict()},
                           args.save)
        print("")
