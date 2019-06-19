from collections import defaultdict

import torch
from torch.nn.functional import normalize

import pygtrie

# leaves out double quote and \\
ASCII = "!#$%&'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[]^_`abcdefghijklmnopqrstuvwxyz{|}~"
ASCII2IDX = {c:i for i, c in enumerate(ASCII)}


def precomp(labe_probs, neighbs, T):
    """
    labe_probs - T x nlabes, normalized along dim 1
    neighbs - nesz length list of tags
    """
    trie = pygtrie.CharTrie()
    labe2idx = ASCII2IDX
    # put all subsequences
    # [trie.__setitem__(''.join(ne[i:j]), True)
    #  for ne in neighbs for i in range(len(ne)) for j in range(i+1, min(len(ne), i+T)+1)]
    [trie.__setitem__(ne[i:j], True)
     for ne in neighbs for i in range(len(ne)) for j in range(i+1, min(len(ne), i+T)+1)]
    # now we can get cost for every start position.
    # Note we'll have trie[pfx][t] = cost of that prefix STARTING AT t
    prev, start = None, None
    for key in trie.iterkeys():
        lastlabe = key[-1]
        if len(key) == 1:
            cost = (1-labe_probs[:, labe2idx[lastlabe]]) # the "cost"
            start = 0
        elif prev is None: # non 1-length restart; figure out where we are
            prev = trie[key[:-1]]
            start = len(key) - 1
            cost = prev[:-1] + (1-labe_probs[start:, labe2idx[lastlabe]])
        else:
            # print(key, start)
            # print(prev)
            cost = prev[:-1] + (1-labe_probs[start:, labe2idx[lastlabe]])
        trie[key] = cost
        if not trie.has_subtrie(key): # the terminal subsequence
            prev, start = None, None
        else:
            prev = cost
            start += 1
    return trie


def dp_single(labe_probs, neighbs, labe2idx, c):
    """
    labe_probs - T x nlabes, normalized along dim 1
    neighbs - nesz length list
    """
    T, nlabes = labe_probs.size()
    # first map labels to ascii strings so this is all simpler
    assert len(labe2idx) <= len(ASCII)
    myneighbs = [''.join([ASCII[labe2idx[labe]] for labe in ne]) for ne in neighbs]
    trie = precomp(labe_probs, myneighbs, T)
    keysbylen = defaultdict(list)
    [keysbylen[len(key)].append(key) for key in trie]
    table = [0]*(T+1)
    #rultable = [0]*(T+1)
    rultable = None
    bps = [None]*(T+1)
    for t in range(1, T+1):
        best_cost, best_choice = float("inf"), None
        for k in range(t):
            suff_len = t-k
            prev_cost = table[k]
            # now we need the best match between x[k:t] and anything of size t-k
            # now find all contiguous spans k:t
            best_k_cost, best_k_choice = float("inf"), None
            for l, labegram in enumerate(keysbylen[suff_len]):
                wrongcost = trie[labegram][k] # cost starting at k
                if wrongcost < best_k_cost:
                    best_k_cost = wrongcost
                    best_k_choice = k, suff_len, l
            if best_k_cost + c + prev_cost < best_cost:
                best_cost = best_k_cost + c + prev_cost
                best_choice = best_k_choice
                #rultable[t] = best_k_cost + rultable[k]
        table[t] = best_cost
        bps[t] = best_choice
    return table, bps, trie, keysbylen, rultable


def backtrack(bps, keysbylen, idx2labe):
    preds = []
    k, suff_len, l = bps[-1]
    preds.append(keysbylen[suff_len][l])
    while k > 0:
        k, suff_len, l = bps[k]
        preds.append(keysbylen[suff_len][l])
    #rulpreds = []
    #rulpreds.extend([pred for pred in preds[::-1]])
    num_copies = len(preds)
    rulpreds = [idx2labe[ASCII2IDX[c]] for pred in reversed(preds) for c in pred]
    return rulpreds, num_copies
