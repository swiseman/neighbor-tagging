import argparse
import torch
import random
import torch.nn as nn
from torch.nn.functional import normalize
from torch.nn.utils.rnn import pad_sequence

from pytorch_pretrained_bert.file_utils import PYTORCH_PRETRAINED_BERT_CACHE
from pytorch_pretrained_bert.optimization import BertAdam, warmup_linear
from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.modeling import BertForTokenClassification, BertConfig, WEIGHTS_NAME, CONFIG_NAME
#from pytorch_pretrained_bert.modeling import BertModel, BertConfig, WEIGHTS_NAME, CONFIG_NAME

import data
import eval_util

def train(sentdb, model, optim, device, args):
    model.train()
    total_loss, total_preds = 0.0, 0
    perm = torch.randperm(len(sentdb.minibatches))
    for i, idx in enumerate(perm):
        optim.zero_grad()
        x, Cx, tgts = sentdb.pp_word_batch(idx.item())
        x, Cx, tgts = x.to(device), Cx.to(device), tgts.to(device)
        bsz, T = x.size()
        mask = (x != 0).long()
        logits = model(x, attention_mask=mask) # bsz x num_wrd_pcs x num_labels I assume
        wlogits = torch.bmm(Cx, logits) # bsz x T x num_labels
        loss = torch.nn.functional.cross_entropy(wlogits.view(-1, wlogits.size(2)),
                                                 tgts.view(-1), reduction='sum')
        total_loss += loss.item()
        loss.div(bsz*T).backward()
        optim.step()
        total_preds += (T*bsz)
        if (i+1) % args.log_interval == 0:
            print("batch", i+1, "loss:", total_loss/total_preds)
    return total_loss/total_preds


def do_fscore(sentdb, model, device, idx2tag, args, scripteval=False, labelmap=None):
    """
    micro-avgd segment-level f1-score
    """
    model.eval()
    total_pred, total_gold, total_crct = 0.0, 0.0, 0.0
    goldss, predss = [], []
    zero_shot = labelmap is not None
    for i in range(len(sentdb.val_minibatches)):
        x, Cx, tgts = sentdb.pp_word_batch(i, val=True, gold_as_str=zero_shot)
        x, Cx = x.to(device), Cx.to(device)
        if not isinstance(tgts, list):
            tgts = tgts.to(device)
        #bsz, T = x.size()
        mask = (x != 0).long()
        logits = model(x, attention_mask=mask) # bsz x num_wrd_pcs x num_labels
        wlogits = torch.bmm(Cx, logits) # bsz x T x num_labels
        _, preds = wlogits.max(2) # bsz x T
        if not zero_shot:
            gold = [[idx2tag[idx.item()] for idx in row] for row in tgts]
        else:
            gold = tgts
        preds = [[idx2tag[idx.item()] for idx in row] for row in preds]
        if labelmap is not None:
            preds = [[labelmap[labe] for labe in pred] for pred in preds]
        if scripteval:
            goldss.append(gold)
            predss.append(preds)
        else:
            if args.acc_eval:
                bpred, bcrct = eval_util.batch_acc_eval(preds, gold)
                bgold = bpred
            else:
                bpred, bgold, bcrct = eval_util.batch_span_eval(preds, gold)
            total_pred += bpred
            total_gold += bgold
            total_crct += bcrct
    if scripteval:
        acc, prec, rec, f1 = eval_util.run_conll(goldss, predss)
        if args.acc_eval:
            microp, micror, microf1 = acc, acc, acc
        else:
            microp, micror, microf1 = prec, rec, f1
    else:
        microp = total_crct/total_pred if total_pred > 0 else 0
        micror = total_crct/total_gold if total_gold > 0 else 0
        microf1 = 2*microp*micror/(microp + micror)
    return microp, micror, microf1

parser = argparse.ArgumentParser()
parser.add_argument("-bert_model", default="bert-base-cased", type=str,
                    choices=["bert-base-uncased", "bert-large-uncased", "bert-base-cased"])
parser.add_argument("-lower", action='store_true', help="")
parser.add_argument("-sent_fi", default="data/conll2003/conll2003-train.words", type=str, help="")
parser.add_argument("-tag_fi", default="data/conll2003/conll2003-train.postags", type=str, help="")
parser.add_argument("-val_sent_fi", default="data/conll2003/conll2003-dev.words", type=str, help="")
parser.add_argument("-val_tag_fi", default="data/conll2003/conll2003-dev.postags", type=str, help="")
parser.add_argument("-db_fi", default="default_sentdb.pt", type=str, help="")
parser.add_argument("-load_saved_db", action='store_true', help="")
parser.add_argument('-bsz', type=int, default=32, help='')
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
parser.add_argument('-epochs', type=int, default=3, help='')
parser.add_argument('-log_interval', type=int, default=100, help='')
parser.add_argument('-save', type=str, default='', help='path to save the final model')

parser.add_argument("-acc_eval", action='store_true', help="")
parser.add_argument("-scripteval", action='store_true', help="")
parser.add_argument('-pred_shard_size', type=int, default=32, help='')
parser.add_argument("-align_strat", type=str, default="first",
                    choices=["first", "last"])
parser.add_argument("-cachedir", type=str, default="/scratch/samstuff/bertmodels")
parser.add_argument("-just_eval", type=str, default=None,
                    choices=["dev", "test", "dev-zero", "test-zero"], help="")

if __name__ == "__main__":
    args = parser.parse_args()
    print(args)
    CACHEDIR = None if args.cachedir.lower() == "none" else args.cachedir

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    #import data # doing this here otherwise randomness gets messed up?

    if torch.cuda.is_available():
        if not args.cuda:
            print("WARNING: You have a CUDA device, so you should probably run with --cuda")
    device = torch.device("cuda" if args.cuda else "cpu")

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
                             parampred=True)
    else:
        sentdb = data.SentDB(args.sent_fi, args.tag_fi, tokenizer, args.val_sent_fi,
                             args.val_tag_fi, lower=args.lower,
                             align_strat=args.align_strat, parampred=True)
        sentdb.make_minibatches(args.bsz, None)
        sentdb.make_minibatches(args.bsz, None, val=True)
        # if args.db_fi is not None:
        #     print("saving db to", args.db_fi)
        #     sentdb.save(args.db_fi)
    idx2tag = sentdb.tagtypes

    if len(args.train_from) > 0:
        print("loading model from", args.train_from)
        saved_stuff = torch.load(args.train_from)
        saved_args = saved_stuff["opt"]
        model = BertForTokenClassification.from_pretrained(
            args.bert_model, num_labels=len(sentdb.tagtypes),
            cache_dir=CACHEDIR)
        model.load_state_dict(saved_stuff["state_dict"])
    else:
        model = BertForTokenClassification.from_pretrained(
            args.bert_model, num_labels=len(sentdb.tagtypes),
            cache_dir=CACHEDIR)

    model = model.to(device)
    model.dropout.p = args.drop


    if args.just_eval is not None:
        import sys
        # pos_c2f = {'ADP': 'IN', 'DET': 'DT', 'NOUN': 'NN', 'NUM': 'CD', '.': ',', 'PRT': 'TO',
        #            'VERB': 'VBD', 'CONJ': 'CC', 'ADV': 'RB', 'PRON': 'PRP', 'ADJ': 'JJ', 'X': 'FW'}
        # labemap = pos_c2f if "zero" in args.just_eval else None
        # also try mapping misc to FAC instead of NORP
        # ner_c2o = {'B-ORG': 'B-ORG', 'I-ORG': 'I-ORG', 'O': 'O', 'B-MISC': 'B-NORP', 'I-MISC': 'I-NORP',
        #            'B-PER': 'B-PERSON', 'I-PER': 'I-PERSON', 'B-LOC': 'B-GPE', 'I-LOC': 'I-GPE'}
        ner_chunk2ner = {'B-NP': 'B-PER', 'B-VP': 'O', 'I-NP': 'I-PER', 'I-VP': 'O', 'O': 'O', 'B-PP': 'O',
                         'B-SBAR': 'O', 'B-ADJP': 'O', 'I-ADJP': 'O', 'B-ADVP': 'O', 'B-PRT': 'O', 'B-CONJP': 'O',
                         'I-CONJP': 'O', 'I-PP': 'O', 'B-INTJ': 'O', 'I-ADVP': 'O', 'B-LST': 'O',
                         'I-SBAR': 'O', 'I-LST': 'O', 'I-INTJ': 'O'}
        #labemap = ner_c2o if "zero" in args.just_eval else None
        labemap = ner_chunk2ner if "zero" in args.just_eval else None
        with torch.no_grad():
            prec, rec, f1 = do_fscore(sentdb, model, device, idx2tag, args,
                                      scripteval=args.scripteval, labelmap=labemap)
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
            prec, rec, f1 = do_fscore(sentdb, model, device, idx2tag, args,
                                      scripteval=args.scripteval)
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
