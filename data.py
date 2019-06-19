"""
this file modified from the word_language_model example
"""
import os
import random
import torch

from collections import Counter, defaultdict

from torch.nn.utils.rnn import pad_sequence
from torch.nn.functional import normalize

class Batch(object):
    def __init__(self, sidx, eidx, neidxs=None, max_crct=None, tgt_nzs=None,
                 tgt_nz_idxs=None, xalign=None, nalign=None):
        self.sidx = sidx
        self.eidx = eidx
        self.neidxs = neidxs
        self.max_crct = max_crct
        self.tgt_nzs = tgt_nzs
        self.tgt_nz_idxs = tgt_nz_idxs
        self.xalign = xalign
        self.nalign = nalign

class SentDB(object):
    def __init__(self, tr_wrdfi, tr_tagfi, tokenizer, val_wrdfi=None, val_tagfi=None,
                 lower=False, path=None, align_strat="last", subsample_all=False,
                 parampred=False):
        self.word_level = True
        self.align_strat = align_strat
        self.subsample = 2500
        self.subsample_all = subsample_all
        self.parampred = parampred
        if path is not None:
            statedict = torch.load(path)
            self.sent_words = statedict["sent_words"]
            self.sent_wpcs = statedict["sent_wpcs"]
            self.sent_tags = statedict["sent_tags"]
            self.minibatches = statedict["minibatches"]
            self.top_nes = statedict["top_nes"]
            self.tag2sent = statedict["tag2sent"]
            if "vsent_words" in statedict:
                self.vsent_words = statedict["vsent_words"]
                self.vsent_wpcs = statedict["vsent_wpcs"]
                self.vsent_tags = statedict["vsent_tags"]
                self.val_minibatches = statedict["val_minibatches"]
                self.vtop_nes = statedict["vtop_nes"]
        else:
            self.sent_words, self.sent_wpcs, self.sent_tags = SentDB.get_wrd_pcs_tags(
                tr_wrdfi, tr_tagfi, tokenizer, lower=lower)
            # get tag2sents just in case we need to supplement
            self.tag2sent = defaultdict(set)
            for i, tags in enumerate(self.sent_tags):
                [self.tag2sent[tag].add(i) for tag in tags]

            if val_wrdfi is not None and val_tagfi is not None:
                self.vsent_words, self.vsent_wpcs, self.vsent_tags = SentDB.get_wrd_pcs_tags(
                    val_wrdfi, val_tagfi, tokenizer, lower=lower)
        tagtypes = set()
        [tagtypes.update(seq) for seq in self.sent_tags]
        self.tagtypes = sorted(tagtypes)
        if self.parampred:
            self.tag2idx = {tt: i for i, tt in enumerate(self.tagtypes)}

    def replace_val_w_test(self, te_wrdfi, te_tagfi, tokenizer, emb_func, device,
                           ne_bsz=128, nne=500, lower=False):
        print("there were", len(self.vsent_words), "val sentences")
        self.vsent_words, self.vsent_wpcs, self.vsent_tags = SentDB.get_wrd_pcs_tags(
            te_wrdfi, te_tagfi, tokenizer, lower=lower)
        assert len(self.vsent_words) == len(self.vsent_wpcs)
        assert len(self.vsent_words) == len(self.vsent_tags)
        print("now there are", len(self.vsent_words), "val sentences")
        self.vtop_nes = None
        print("recomputing neighbors...")
        trembs = self.get_all_embs(ne_bsz, emb_func, device, cosine=True)
        vembs = self.get_all_embs(ne_bsz, emb_func, device, cosine=True, val=True)
        with torch.no_grad():
            G = vembs.mm(trembs.t())
            _, argtop = torch.topk(G, nne, dim=1) # these are sorted
            self.vtop_nes = [[idx.item() for idx in row] for row in argtop]

    @staticmethod
    def get_wrd_pcs_tags(wrdfi, tagfi, tokenizer, lower=False):
        sent_words, sent_wpcs, sent_tags = [], [], []
        with open(wrdfi) as f1:
            with open(tagfi) as f2:
                for line in f1:
                    sent = line.strip()
                    # N.B. don't need to lower, since tokenizer does it automatically
                    # if lower:
                    #     sent = sent.lower()
                    words = sent.split()
                    tags = f2.readline().strip().split()
                    assert len(tags) == len(words)
                    wpcs = ["[CLS]"]
                    wpcs.extend(tokenizer.tokenize(sent))
                    wpcs.append("[SEP]")
                    try:
                        aligns = SentDB.align_wpcs(words, wpcs, lower=lower)
                    except AssertionError:
                        print("ignoring one from", wrdfi)
                        continue
                    sent_wpcs.append(tokenizer.convert_tokens_to_ids(wpcs))
                    sent_words.append(aligns)
                    sent_tags.append(tags)
        # shuffle before sorting by length
        perm = [t.item() for t in torch.randperm(len(sent_words))]
        perm.sort(key=lambda idx: len(sent_words[idx]))
        sent_words = [sent_words[idx] for idx in perm]
        sent_wpcs = [sent_wpcs[idx] for idx in perm]
        sent_tags = [sent_tags[idx] for idx in perm]
        return sent_words, sent_wpcs, sent_tags

    @staticmethod
    def align_wpcs(words, wpcs, lower=False):
        """
        maps each word idx to start and end idx w/ in wpcs.
        assumes wpcs is padded on either end with CLS and SEP
        """
        align = []
        curr_start, curr_wrd = 1, 0 # start at 1, b/c of CLS
        buf = []
        for i in range(1, len(wpcs)-1): # ignore [SEP] final token
            strpd = wpcs[i][2:] if wpcs[i].startswith("##") else wpcs[i]
            buf.append(strpd)
            #buf.append(wpcs[i].lstrip('##'))
            fwrd = ''.join(buf)
            wrd = words[curr_wrd].lower() if lower else words[curr_wrd]
            if fwrd == wrd or fwrd == "[UNK]":
                align.append((curr_start, i+1))
                curr_start = i+1
                curr_wrd += 1
                buf = []
        assert curr_wrd == len(words)
        return align

    def get_all_embs(self, bsz, emb_func, device, cosine=True, val=False):
        if val:
            sent_words, sent_wpcs = self.vsent_words, self.vsent_wpcs
        else:
            sent_words, sent_wpcs = self.sent_words, self.sent_wpcs
        curr_len, start = len(sent_words[0]), 0
        all_embs = []
        with torch.no_grad():
            for i in range(len(sent_words)):
                if len(sent_words[i]) != curr_len or i-start == bsz: # we're done
                    #batch = torch.LongTensor([self.sent_wpcs[j] ])
                    batch = pad_sequence([torch.LongTensor(sent_wpcs[j])  # max_wpcs x bsz
                                          for j in range(start, i)], padding_value=0)
                    embs = emb_func(batch.t().to(device)) # bsz x emb_size
                    all_embs.append(embs)
                    curr_len, start = len(sent_words[i]), i
            if len(sent_words) > start:
                batch = pad_sequence([torch.LongTensor(sent_wpcs[j])  # max_wpcs x bsz
                                      for j in range(start, len(sent_words))],
                                     padding_value=0)
                embs = emb_func(batch.t().to(device)) # bsz x emb_size
                all_embs.append(embs)
            all_embs = torch.cat(all_embs, 0)
            assert all_embs.size(0) == len(sent_words)
            if cosine:
                all_embs = normalize(all_embs, p=2, dim=1)
        return all_embs

    def compute_top_neighbs(self, bsz, emb_func, nne, device, cosine=True, ignore_trne=False):
        trembs = self.get_all_embs(bsz, emb_func, device, cosine=cosine)
        if ignore_trne:
            self.top_nes = None
        else:
            with torch.no_grad():
                G = trembs.mm(trembs.t())
                rng = torch.arange(trembs.size(0))
                G[rng, rng] = 0 # set diagonal to zero
                _, argtop = torch.topk(G, nne, dim=1) # these are sorted
                self.top_nes = [[idx.item() for idx in row] for row in argtop]
            assert len(self.top_nes) == len(self.sent_words)
        if hasattr(self, "vsent_words"):
            vembs = self.get_all_embs(bsz, emb_func, device, cosine=cosine, val=True)
            with torch.no_grad():
                G = vembs.mm(trembs.t())
                _, argtop = torch.topk(G, nne, dim=1) # these are sorted
                self.vtop_nes = [[idx.item() for idx in row] for row in argtop]

    def pp_word_batch(self, batch_idx, padidx=0, val=False, gold_as_str=False):
        if val:
            sent_tags, sent_wpcs = self.vsent_tags, self.vsent_wpcs
            minibatches = self.val_minibatches
        else:
            sent_tags, sent_wpcs = self.sent_tags, self.sent_wpcs
            minibatches = self.minibatches
        batch = minibatches[batch_idx]
        x = pad_sequence([torch.LongTensor(sent_wpcs[i])  # max_wpcs x bsz
                          for i in range(batch.sidx, batch.eidx)], padding_value=padidx)
        if gold_as_str:
            gold = [sent_tags[i] for i in range(batch.sidx, batch.eidx)]
        else:
            # # this is just for Chunking which is broken
            # gold = torch.stack([torch.LongTensor([self.tag2idx[tag] if tag in self.tag2idx else self.tag2idx['O'] for tag in sent_tags[i]])
            #                     for i in range(batch.sidx, batch.eidx)])
            gold = torch.stack([torch.LongTensor([self.tag2idx[tag] for tag in sent_tags[i]])
                                for i in range(batch.sidx, batch.eidx)])
        #T = gold.size(1)
        T = len(sent_tags[batch.sidx])
        bsz = batch.eidx - batch.sidx
        Cx = torch.zeros(bsz, T, x.size(0))
        xalign = self.get_align_nnz([idx for idx in range(batch.sidx, batch.eidx)], val=val)
        Cx.view(-1).scatter_(0, torch.LongTensor(xalign), 1)
        return x.t(), Cx, gold

    def word_batch(self, batch_idx, padidx=0, val=False):
        if val:
            sent_words, sent_wpcs = self.vsent_words, self.vsent_wpcs
            minibatches = self.val_minibatches
        else:
            sent_words, sent_wpcs = self.sent_words, self.sent_wpcs
            minibatches = self.minibatches
        batch = minibatches[batch_idx]
        if batch.neidxs is None: # we're doing this randomly
            assert not val
            batch = self.precompute_word_batch(batch.sidx, batch.eidx, batch.max_crct,
                                               random_ne=True, val=False)
        x = pad_sequence([torch.LongTensor(sent_wpcs[i])  # max_wpcs x bsz
                          for i in range(batch.sidx, batch.eidx)], padding_value=padidx)
        neighbs = pad_sequence([torch.LongTensor(self.sent_wpcs[i]) for i in batch.neidxs],
                               padding_value=padidx) # max_wpcs x nesz
        Tn = max(len(self.sent_tags[i]) for i in batch.neidxs)
        nne = len(batch.neidxs)
        T = len(sent_words[batch.sidx])
        bsz = batch.eidx - batch.sidx
        Cx = torch.zeros(bsz, T, x.size(0))
        Cn = torch.zeros(nne, Tn, neighbs.size(0))
        # ugh have to fill these in somehow
        Cx.view(-1).scatter_(0, batch.xalign.long(), 1)
        Cn.view(-1).scatter_(0, batch.nalign.long(), 1)
        # neighb tgts are in format nne*max_ne_len, so add one more option for ignore
        ignoreidx = nne*Tn
        tgts = torch.LongTensor(bsz, T, batch.max_crct).fill_(ignoreidx)
        tgts.view(-1).scatter_(0, batch.tgt_nz_idxs.long(), batch.tgt_nzs.long())
        return x.t(), neighbs.t(), Cx, Cn, tgts

    def predword_batch(self, batch_idx, ne_per_sent, padidx=0):
        """
        makes a val minibatch for actually predicting.
        N.B. only uses val minibatches
        """
        sent_words, sent_wpcs, sent_tags = self.vsent_words, self.vsent_wpcs, self.vsent_tags
        minibatches = self.val_minibatches
        batch = minibatches[batch_idx]
        x = pad_sequence([torch.LongTensor(sent_wpcs[i])  # max_wpcs x bsz
                          for i in range(batch.sidx, batch.eidx)], padding_value=padidx)
        gold = [sent_tags[i] for i in range(batch.sidx, batch.eidx)]
        T = len(sent_words[batch.sidx])
        bsz = batch.eidx - batch.sidx
        Cx = torch.zeros(bsz, T, x.size(0))
        Cx.view(-1).scatter_(0, batch.xalign.long(), 1)

        # make new neighbors so we can have more than we train with; kinda stupid
        neidxs = [neidx for xidx in range(batch.sidx, batch.eidx)
                  for neidx in self.vtop_nes[xidx][:ne_per_sent]]

        # also add a neighbor for every missing tag
        netags = set(self.sent_tags[neidxs[0]])
        [netags.update(self.sent_tags[neidx]) for neidx in neidxs[1:]]

        for tag, idxset in self.tag2sent.items():
            if tag not in netags:
                kosher = list(idxset)
                assert len(kosher) > 0
                ridx = torch.randint(0, len(kosher), (1,))[0].item()
                neidxs.append(kosher[ridx])

        # map each tag to location in neighbors
        tag2nes = defaultdict(list)
        [tag2nes[self.sent_tags[neidx][i]].append((n, i))
         for n, neidx in enumerate(neidxs) for i in range(len(self.sent_tags[neidx]))]

        # just hacking for now
        subkeys = ['O']
        if self.subsample_all:
            subkeys.extend(tag for tag in tag2nes.keys() if tag != 'O')
        for key in subkeys:
            if key in tag2nes and len(tag2nes[key]) > self.subsample:
                perm = torch.randperm(len(tag2nes[key]))
                tag2nes[key] = [tag2nes[key][idx.item()] for idx in perm[:self.subsample]]

        nne, Tn = len(neidxs), max(len(self.sent_tags[neidx]) for neidx in neidxs)
        assert len(tag2nes) == len(self.tagtypes)
        tag2mask = []
        for tag in self.tagtypes: # zeros only for indices w/ corresponding tag
            mask = torch.Tensor(nne*Tn).fill_(-float("inf"))
            #zidxs = torch.LongTensor([i*nne + n for (n, i) in tag2nes[tag]])
            zidxs = torch.LongTensor([n*Tn + i for (n, i) in tag2nes[tag]])
            mask.scatter_(0, zidxs, 0)
            tag2mask.append((tag, mask.view(1, -1))) # 1 x nne*Tn

        # finally make ne stuff
        neighbs = pad_sequence([torch.LongTensor(self.sent_wpcs[i]) for i in neidxs],
                               padding_value=padidx) # max_wpcs x nesz
        nalign = self.get_align_nnz(neidxs)
        Cn = torch.zeros(nne, Tn, neighbs.size(0))
        Cn.view(-1).scatter_(0, torch.LongTensor(nalign), 1)
        return x.t(), neighbs.t(), Cx, Cn, tag2mask, gold


    def pred_single_batch(self, sidx, ne_per_sent, padidx=0):
        """
        makes a val minibatch for actually predicting.
        N.B. only uses val minibatches
        """
        sent_words, sent_wpcs, sent_tags = self.vsent_words, self.vsent_wpcs, self.vsent_tags
        x = torch.LongTensor(sent_wpcs[sidx]).view(1, -1) # 1 x whatever
        gold = [sent_tags[sidx]]
        T = len(sent_words[sidx])
        bsz = 1
        Cx = torch.zeros(bsz, T, x.size(1))
        xalign = self.get_align_nnz([sidx], val=True)
        Cx.view(-1).scatter_(0, torch.LongTensor(xalign), 1)

        neidxs = self.vtop_nes[sidx][:ne_per_sent]
        # also add a neighbor for every missing tag
        netags = set(self.sent_tags[neidxs[0]])
        [netags.update(self.sent_tags[neidx]) for neidx in neidxs[1:]]

        for tag, idxset in self.tag2sent.items():
            if tag not in netags:
                kosher = list(idxset)
                assert len(kosher) > 0
                ridx = torch.randint(0, len(kosher), (1,))[0].item()
                neidxs.append(kosher[ridx])

        # map each tag to location in neighbors
        tag2nes = defaultdict(list)
        [tag2nes[self.sent_tags[neidx][i]].append((n, i))
         for n, neidx in enumerate(neidxs) for i in range(len(self.sent_tags[neidx]))]


        nne, Tn = len(neidxs), max(len(self.sent_tags[neidx]) for neidx in neidxs)
        assert len(tag2nes) == len(self.tagtypes)
        tag2mask = []
        for tag in self.tagtypes: # zeros only for indices w/ corresponding tag
            mask = torch.Tensor(nne*Tn).fill_(-float("inf"))
            zidxs = torch.LongTensor([n*Tn + i for (n, i) in tag2nes[tag]])
            mask.scatter_(0, zidxs, 0)
            tag2mask.append((tag, mask.view(1, -1))) # 1 x nne*Tn

        # finally make ne stuff
        neighbs = pad_sequence([torch.LongTensor(self.sent_wpcs[i]) for i in neidxs],
                               padding_value=padidx) # max_wpcs x nesz
        nalign = self.get_align_nnz(neidxs)
        Cn = torch.zeros(nne, Tn, neighbs.size(0))
        Cn.view(-1).scatter_(0, torch.LongTensor(nalign), 1)
        ne_tag_seqs = [self.sent_tags[neidx] for neidx in neidxs]
        return x, neighbs.t(), Cx, Cn, tag2mask, gold, ne_tag_seqs


    def make_minibatches(self, bsz, ne_per_sent, random_tr_ne=False, val=False):
        if val:
            sent_words = self.vsent_words
            random_tr_ne = False
        else:
            sent_words = self.sent_words
        curr_len, start = len(sent_words[0]), 0
        minibatches = []
        for i in range(len(sent_words)):
            if len(sent_words[i]) != curr_len or i-start == bsz: # we're done
                if self.word_level:
                    if random_tr_ne or self.parampred:
                        batch = Batch(start, i, max_crct=ne_per_sent) # HACK!
                    else:
                        batch = self.precompute_word_batch(start, i, ne_per_sent, val=val)
                    minibatches.append(batch)
                else:
                    pass
                curr_len, start = len(sent_words[i]), i
        # catch last
        if len(sent_words) > start:
            if self.word_level:
                if random_tr_ne or self.parampred:
                    batch = Batch(start, len(sent_words), max_crct=ne_per_sent) # HACK!
                else:
                    batch = self.precompute_word_batch(
                        start, len(sent_words), ne_per_sent, val=val)
                minibatches.append(batch)
            else:
                pass
        if val:
            self.val_minibatches = minibatches
        else:
            self.minibatches = minibatches

    def get_align_nnz(self, batch, val=False):
        # calculate word to word piece alignment for everybody:
        # Cx will be bsz x T x max_batch_wrdpieces
        # Cn will be nne x max_ne_len x max_ne_wrdpieces
        if val:
            sent_words, sent_wpcs = self.vsent_words, self.vsent_wpcs
        else:
            sent_words, sent_wpcs = self.sent_words, self.sent_wpcs
        max_wpcs = max(len(sent_wpcs[idx]) for idx in batch)
        max_wrds = max(len(sent_words[idx]) for idx in batch)
        if self.align_strat == "sum":
            nz_idxs = [b*max_wrds*max_wpcs + t*max_wpcs + i
                       for b, idx in enumerate(batch)
                       for t in range(len(sent_words[idx]))
                       for i in range(*sent_words[idx][t])]
        elif self.align_strat == "first":
            nz_idxs = [b*max_wrds*max_wpcs + t*max_wpcs + sent_words[idx][t][0]
                       for b, idx in enumerate(batch)
                       for t in range(len(sent_words[idx]))]
        else: # last
            nz_idxs = [b*max_wrds*max_wpcs + t*max_wpcs + sent_words[idx][t][1]-1
                       for b, idx in enumerate(batch)
                       for t in range(len(sent_words[idx]))]
        return nz_idxs

    def precompute_word_batch(self, sidx, eidx, ne_per_sent, random_ne=False, val=False):
        if val:
            sent_tags, top_nes = self.vsent_tags, self.vtop_nes
        else:
            sent_tags, top_nes = self.sent_tags, self.top_nes
        used = set(range(sidx, eidx)) if not val else set()
        if random_ne:
            neidxs = [neidx for xidx in range(sidx, eidx)
                      for neidx in random.sample(top_nes[xidx], ne_per_sent)
                      if neidx not in used]
        else:
            neidxs = [neidx for xidx in range(sidx, eidx)
                      for neidx in top_nes[xidx][:ne_per_sent]
                      if neidx not in used]

        # also add a neighbor for every missing tag
        netags = set(self.sent_tags[neidxs[0]])
        [netags.update(self.sent_tags[neidx]) for neidx in neidxs[1:]]

        for tag, idxset in self.tag2sent.items():
            if tag not in netags:
                kosher = [idx for idx in idxset if idx not in used]
                assert len(kosher) > 0
                ridx = torch.randint(0, len(kosher), (1,))[0].item()
                neidxs.append(kosher[ridx])

        # map each tag to location in neighbors
        tag2nes = defaultdict(list)
        [tag2nes[self.sent_tags[neidx][i]].append((n, i))
         for n, neidx in enumerate(neidxs) for i in range(len(self.sent_tags[neidx]))]

        # just hacking for now
        subkeys = ['O']
        if self.subsample_all:
            subkeys.extend(tag for tag in tag2nes.keys() if tag != 'O')

        #if 'O' in tag2nes and len(tag2nes['O']) > self.subsample:
        #    perm = torch.randperm(len(tag2nes['O']))
        #    tag2nes['O'] = [tag2nes['O'][idx.item()] for idx in perm[:self.subsample]]
        for key in subkeys:
            if key in tag2nes and len(tag2nes[key]) > self.subsample:
                perm = torch.randperm(len(tag2nes[key]))
                tag2nes[key] = [tag2nes[key][idx.item()] for idx in perm[:self.subsample]]

        max_crct = max(len(tag2nes[tag]) for b in range(sidx, eidx) for tag in sent_tags[b]
                       if tag in tag2nes)
        nne = len(neidxs)
        T, bsz = len(sent_tags[sidx]), eidx - sidx
        # neighb words will be in format max_len*nne; tgt tensor will be T x bsz x max_crct
        # neighb words will be in format nne*max_len; tgt tensor will be bsz x T x max_crct
        Tn = max(len(self.sent_tags[neidx]) for neidx in neidxs)
        tgt_nzs, tgt_nz_idxs = [], []
        for t in range(T):
            for b in range(sidx, eidx):
                trutag = sent_tags[b][t]
                assert trutag in tag2nes
                crcts = [n*Tn + i for (n, i) in tag2nes[trutag]]
                tgt_nz_idxs.extend([(b-sidx)*T*max_crct + t*max_crct + r
                                    for r in range(len(crcts))])
                tgt_nzs.extend(crcts)
        xalign = self.get_align_nnz([idx for idx in range(sidx, eidx)], val=val)
        nalign = self.get_align_nnz(neidxs)
        batch = Batch(sidx, eidx, neidxs, max_crct, torch.IntTensor(tgt_nzs),
                      torch.IntTensor(tgt_nz_idxs), torch.IntTensor(xalign),
                      torch.IntTensor(nalign))

        return batch

    def save(self, path):
        state = {"sent_words": self.sent_words, "sent_wpcs": self.sent_wpcs,
                 "sent_tags": self.sent_tags, "minibatches": self.minibatches,
                 "top_nes": self.top_nes, "tag2sent": self.tag2sent}
        if hasattr(self, "vsent_words"):
            state.update({"vsent_words": self.vsent_words, "vsent_wpcs": self.vsent_wpcs,
                          "vsent_tags": self.vsent_tags, "val_minibatches": self.val_minibatches,
                          "vtop_nes": self.vtop_nes})
        torch.save(state, path)
