import argparse
import torch
import random
import torch.nn as nn
from torch.nn.functional import normalize
from torch.nn.utils.rnn import pad_sequence

from pytorch_pretrained_bert.file_utils import PYTORCH_PRETRAINED_BERT_CACHE
from pytorch_pretrained_bert.optimization import BertAdam, warmup_linear
from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.modeling import BertModel, BertConfig, WEIGHTS_NAME, CONFIG_NAME

import data
import eval_util

from dp_pred import dp_single, backtrack

class TagWordModel(nn.Module):
    def __init__(self, opt):
        super(TagWordModel, self).__init__()
        self.register_buffer('dummy', torch.Tensor(1, 1).fill_(-float("inf")))
        self.bert = BertModel.from_pretrained(opt.bert_model, cache_dir=CACHEDIR)
        for lay in self.bert.encoder.layer:
            lay.output.dropout.p = args.drop

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
                bertrep, _ = self.bert(xsplit, attention_mask=mask,
                                       output_all_encoded_layers=False)
                splitrep = torch.bmm(Csplits[i], bertrep) # shardsz x T x dim
                wreps.append(splitrep) # shardsz x T x dim
            wreps = torch.cat(wreps, 0) # bsz x T x dim
        else:
            mask = (x != 0).long()
            bertrep, _ = self.bert(x, attention_mask=mask,
                                   output_all_encoded_layers=False) # bsz x num_pcs x dim
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

def train(sentdb, model, optim, device, args):
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
        total_loss += get_batch_loss(batch_reps, ne_reps.view(-1, ne_reps.size(2)),
                                     tgts, model.dummy, args)
        optim.step()
        total_preds += (T*bsz)
        if (i+1) % args.log_interval == 0:
            print("batch", i+1, "loss:", total_loss/total_preds)
    return total_loss/total_preds


def do_fscore(sentdb, model, device, args):
    """
    micro-avgd segment-level f1-score
    """
    model.eval()
    total_pred, total_gold, total_crct = 0.0, 0.0, 0.0
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
                                tag2mask, args) # bsz x T
        if args.acc_eval:
            bpred, bcrct = eval_util.batch_acc_eval(preds, gold)
            bgold = bpred
        else:
            bpred, bgold, bcrct = eval_util.batch_span_eval(preds, gold)
        total_pred += bpred
        total_gold += bgold
        total_crct += bcrct
    microp = total_crct/total_pred if total_pred > 0 else 0
    micror = total_crct/total_gold if total_gold > 0 else 0
    microf1 = 2*microp*micror/(microp + micror)
    return microp, micror, microf1


# the point of this is so we don't mix neighbors based on other stuff in the minibatch
def do_single_fscore(sentdb, model, device, args):
    """
    micro-avgd segment-level f1-score
    """
    model.eval()
    total_pred, total_gold, total_crct = 0.0, 0.0, 0.0
    total_copies, total_words = 0, 0
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
        if ncopies is not None:
            total_copies += ncopies
        if args.acc_eval:
            bpred, bcrct = eval_util.batch_acc_eval(preds, gold)
            bgold = bpred
        else:
            bpred, bgold, bcrct = eval_util.batch_span_eval(preds, gold)
        total_pred += bpred
        total_gold += bgold
        total_crct += bcrct
        total_words += Cx.size(1)
    if args.dp_pred:
        print("avg moves/sent", total_copies/len(sentdb.vsent_words))
        print("avg words/move", total_words/total_copies)
    microp = total_crct/total_pred if total_pred > 0 else 0
    micror = total_crct/total_gold if total_gold > 0 else 0
    microf1 = 2*microp*micror/(microp + micror)
    return microp, micror, microf1


parser = argparse.ArgumentParser()
parser.add_argument("-bert_model", default="bert-base-cased", type=str,
                    choices=["bert-base-uncased", "bert-large-uncased", "bert-base-cased"])
parser.add_argument("-lower", action='store_true', help="")
parser.add_argument("-sent_fi", default="data/ptb/ptb-train.words", type=str, help="")
parser.add_argument("-tag_fi", default="data/ptb/ptb-train.tags", type=str, help="")
parser.add_argument("-val_sent_fi", default="data/ptb/ptb-dev.words", type=str, help="")
parser.add_argument("-val_tag_fi", default="data/ptb/ptb-dev.tags", type=str, help="")
parser.add_argument("-db_fi", default="default_sentdb.pt", type=str, help="")
parser.add_argument("-load_saved_db", action='store_true', help="")
parser.add_argument('-bsz', type=int, default=16, help='')
parser.add_argument('-seed', type=int, default=3435, help='')
parser.add_argument("-cuda", action='store_true', help="")
parser.add_argument("-train_from", default="", type=str, help="")
parser.add_argument("-nosplit_parenth", action='store_true', help="")

parser.add_argument("-lr", default=5e-5, type=float, help="The initial learning rate for Adam.")
parser.add_argument("-drop", default=0.1, type=float, help="")
parser.add_argument("-clip", default=1, type=float, help="")
parser.add_argument("-warmup_prop", default=0.1, type=float,
                    help="Proportion of training to perform linear learning rate warmup for.  "
                         "E.g., 0.1 = 10%% of training.")
parser.add_argument('-epochs', type=int, default=40, help='')
parser.add_argument('-log_interval', type=int, default=100, help='')
parser.add_argument('-save', type=str, default='', help='path to save the final model')

parser.add_argument('-ne_per_sent', type=int, default=20, help='')
parser.add_argument("-random_tr_ne", action='store_true', help="")
parser.add_argument('-eval_ne_per_sent', type=int, default=100, help='')
parser.add_argument("-acc_eval", action='store_true', help="")
parser.add_argument('-pred_shard_size', type=int, default=32, help='')
parser.add_argument("-cosine", action='store_true', help="")
parser.add_argument("-align_strat", type=str, default="first",
                    choices=["sum", "first", "last"])
parser.add_argument("-detach_db", action='store_true', help="")
parser.add_argument("-subsample_all", action='store_true', help="")
parser.add_argument("-cachedir", type=str, default="/scratch/samstuff/bertmodels")
parser.add_argument("-just_eval", type=str, default=None,
                    choices=["dev", "test", "dev-newne", "test-newne"], help="")
parser.add_argument("-zero_shot", action='store_true', help="")

parser.add_argument("-dp_pred", action='store_true', help="")
parser.add_argument("-c", default=1, type=float, help="")

if __name__ == "__main__":
    args = parser.parse_args()
    print(args)
    CACHEDIR = None if args.cachedir.lower() == "none" else args.cachedir

    torch.set_num_threads(2)
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    if args.zero_shot:
        assert not args.load_saved_db
        assert args.just_eval is not None

    if torch.cuda.is_available():
        if not args.cuda:
            print("WARNING: You have a CUDA device, so you should probably run with --cuda")
    device = torch.device("cuda" if args.cuda else "cpu")

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
    tokenizer = BertTokenizer.from_pretrained(
        args.bert_model, do_lower_case=args.lower, never_split=nosplits,
        cache_dir=CACHEDIR)
    print("make sure you agree with never_splits!")

    if args.load_saved_db and args.db_fi is not None:
        print("loading db from", args.db_fi)
        sentdb = data.SentDB(None, None, None, path=args.db_fi, align_strat=args.align_strat,
                             subsample_all=args.subsample_all)
    else:
        sentdb = data.SentDB(args.sent_fi, args.tag_fi, tokenizer, args.val_sent_fi,
                             args.val_tag_fi, lower=args.lower,
                             align_strat=args.align_strat, subsample_all=args.subsample_all)
        if args.db_fi is not None:
            print("saving db to", args.db_fi)
            sentdb.save(args.db_fi)

        nebert = model.bert
        if args.zero_shot and "newne" not in args.just_eval:
            nebert = BertModel.from_pretrained(args.bert_model, cache_dir=CACHEDIR)
            nebert = nebert.to(device)
        def avg_bert_emb(x):
            mask = (x != 0)
            rep, _ = nebert(x, attention_mask=mask.long(), output_all_encoded_layers=False)
            mask = mask.float().unsqueeze(2) # bsz x T x 1
            avgs = (rep * mask).sum(1) / mask.sum(1) # bsz x hid
            return avgs

        nebsz, nne = 128, 500
        # we always compute neighbors w/ cosine; seems to be a bit better
        model.eval()
        sentdb.compute_top_neighbs(nebsz, avg_bert_emb, nne, device, cosine=True,
                                   ignore_trne=args.zero_shot)
        if not args.zero_shot:
            sentdb.make_minibatches(args.bsz, args.ne_per_sent, random_tr_ne=args.random_tr_ne)
            sentdb.make_minibatches(args.bsz, args.ne_per_sent, val=True)
            if args.db_fi is not None:
                print("saving db to", args.db_fi)
                sentdb.save(args.db_fi)
            model.train()

    if args.just_eval is not None:
        import sys

        if not args.zero_shot and args.just_eval != "dev": # if zero_shot we already did this
            # need to recompute new neighbors
            if "newne" in args.just_eval:
                bert = model.bert
            else:
                bert = BertModel.from_pretrained(args.bert_model, cache_dir=CACHEDIR)
                bert = bert.to(device)
            bert.eval()

            def newne_avg_bert_emb(x):
                mask = (x != 0)
                rep, _ = bert(x, attention_mask=mask.long(), output_all_encoded_layers=False)
                mask = mask.float().unsqueeze(2) # bsz x T x 1
                avgs = (rep * mask).sum(1) / mask.sum(1) # bsz x hid
                return avgs

            # note that ne_bsz and nne are just used for recomputing neighbors
            # and for the max ne stored per sent, resp. we control how many ne
            # are used at prediction time w/ eval_ne_per_sent
            sentdb.replace_val_w_test(args.val_sent_fi, args.val_tag_fi, tokenizer,
                                      newne_avg_bert_emb, device,
                                      ne_bsz=128, nne=500, lower=args.lower)
        with torch.no_grad():
            prec, rec, f1 = do_single_fscore(sentdb, model, device, args)
            print("Eval: | P: {:3.5f} / R: {:3.5f} / F: {:3.5f}".format(
                prec, rec, f1))
            sys.exit(0)

    # stuff copied from huggingface
    # Prepare optimizer
    param_optimizer = list(model.named_parameters())

    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer
                    if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer
                    if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}]
    num_train_steps = args.epochs*len(sentdb.minibatches)
    # there's a ton of other options like grad clipping, momentum stuff, schedules, etc
    optimizer = BertAdam(optimizer_grouped_parameters, lr=args.lr, warmup=args.warmup_prop,
                         t_total=num_train_steps, max_grad_norm=args.clip)
    #assert False

    bestloss = float("inf")
    for ep in range(args.epochs):
        trloss = train(sentdb, model, optimizer, device, args)
        print("Epoch {:3d} | train loss {:8.3f}".format(ep, trloss))
        # with torch.no_grad():
        #     valloss = validate(sentdb, model, device, args)
        # print("Epoch {:3d} | val loss {:8.3f}".format(ep, valloss))
        with torch.no_grad():
            prec, rec, f1 = do_fscore(sentdb, model, device, args)
            print("Epoch {:3d} | P: {:3.5f} / R: {:3.5f} / F: {:3.5f}".format(
                ep, prec, rec, f1))
            valloss = -f1
        if valloss < bestloss:
            bestloss = valloss
            if len(args.save) > 0:
                print("saving to", args.save)
                torch.save({"opt": args, "state_dict": model.state_dict()},
                           args.save)
        print("")
