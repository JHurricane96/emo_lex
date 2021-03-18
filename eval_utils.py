# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import io
import numpy as np
import collections

def unit_norm(x):
    norm = np.linalg.norm(x, axis=1)
    norm[norm == 0] = 1
    return x / norm[:, np.newaxis]

def load_vectors(fname, maxload=200000, norm=True, center=False, verbose=True):
    if verbose:
        print("Loading vectors from %s" % fname)
    fin = io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
    n, d = map(int, fin.readline().split())
    if maxload > 0:
        n = min(n, maxload)
    x = np.zeros([n, d])
    words = []
    for i, line in enumerate(fin):
        if i >= n:
            break
        tokens = line.rstrip().split(' ')
        words.append(tokens[0])
        v = np.array(tokens[1:], dtype=float)
        x[i, :] = v
    if norm:
        # x /= np.linalg.norm(x, axis=1)[:, np.newaxis] + 1e-8
        x = unit_norm(x)
    if center:
        x -= x.mean(axis=0)[np.newaxis, :]
        # x /= np.linalg.norm(x, axis=1)[:, np.newaxis] + 1e-8
        x = unit_norm(x)
    if verbose:
        print("%d word vectors loaded" % (len(words)))
    return words, x

def idx(words):
    w2i = {}
    for i, w in enumerate(words):
        if w not in w2i:
            w2i[w] = i
    return w2i

def load_lexicon(filename, words_src, words_tgt, verbose=True):
    f = io.open(filename, 'r', encoding='utf-8')
    lexicon = collections.defaultdict(set)
    idx_src , idx_tgt = idx(words_src), idx(words_tgt)
    vocab = set()
    for line in f:
        whitespace_parts = line.strip().split()
        tab_parts = line.strip().split("\t")
        if len(whitespace_parts) > 2 and len(tab_parts) == 2:
            word_src, word_tgt = line.strip().split("\t")
        else:
            word_src, word_tgt = line.strip().split()
        if word_src in idx_src and word_tgt in idx_tgt:
            lexicon[idx_src[word_src]].add(idx_tgt[word_tgt])
        vocab.add(word_src)
    if verbose:
        coverage = len(lexicon) / float(len(vocab))
        print("Coverage of source vocab: %.4f" % (coverage))
    return lexicon, float(len(vocab))

def compute_nn_accuracy(x_src, x_tgt, lexicon, acc_at=1, bsz=100, lexicon_size=-1):
    if lexicon_size < 0:
        lexicon_size = len(lexicon)
    idx_src = list(lexicon.keys())
    acc = 0.0
    x_src /= np.linalg.norm(x_src, axis=1)[:, np.newaxis] + 1e-8
    x_tgt /= np.linalg.norm(x_tgt, axis=1)[:, np.newaxis] + 1e-8
    for i in range(0, len(idx_src), bsz):
        e = min(i + bsz, len(idx_src))
        scores = np.dot(x_tgt, x_src[idx_src[i:e]].T)
        # pred = scores.argmax(axis=0)
        pred = scores.argpartition(acc_at, axis=0)[-acc_at:]
        for j in range(i, e):
            # if pred[j - i] in lexicon[idx_src[j]]:
            if any(int(p) in lexicon[idx_src[j]] for p in pred[:, j-i]):
                acc += 1.0
    return acc / lexicon_size

def compute_csls_scores(x_src, x_tgt, idx_src, k=10, bsz=1024):
    x_src /= np.linalg.norm(x_src, axis=1)[:, np.newaxis] + 1e-8
    x_tgt /= np.linalg.norm(x_tgt, axis=1)[:, np.newaxis] + 1e-8

    sr = x_src[list(idx_src)]
    sc = np.dot(sr, x_tgt.T)
    similarities = 2 * sc
    sc2 = np.zeros(x_tgt.shape[0])
    for i in range(0, x_tgt.shape[0], bsz):
        j = min(i + bsz, x_tgt.shape[0])
        sc_batch = np.dot(x_tgt[i:j, :], x_src.T)
        dotprod = np.partition(sc_batch, -k, axis=1)[:, -k:]
        sc2[i:j] = np.mean(dotprod, axis=1)
    similarities -= sc2[np.newaxis, :]

    # nn = np.argmax(similarities, axis=1).tolist()
    return similarities

def compute_csls_maps(x_src, words_src, x_tgt, lexicon, nn_words, acc_at=1, lexicon_size=-1, k=10, bsz=1024):
    # idx_src = list(idx(words_src).values())
    idx_map = idx(words_src)
    idx_src = []
    if nn_words:
        idx_src = [idx for word, idx in idx_map.items() if word in nn_words]
    else:
        idx_src = list(idx_map.values())
    similarities = compute_csls_scores(x_src, x_tgt, idx_src, k=k, bsz=bsz)
    nn = np.argpartition(-similarities, range(acc_at), axis=1)[:, :acc_at]
    max_scores = np.take_along_axis(similarities, nn, axis=1)
    map = {}
    print(len(nn))
    for k in range(0, len(nn)):
        correct = 0
        if lexicon != None:
            if idx_src[k] not in lexicon:
                correct = 1
            elif any(int(w) in lexicon[idx_src[k]] for w in nn[k]):
                correct = 2
        map[idx_src[k]] = (nn[k], max_scores[k], correct)
    return map

def compute_csls_accuracy(x_src, x_tgt, lexicon, acc_at=1, lexicon_size=-1, k=10, bsz=1024):
    if lexicon_size < 0:
        lexicon_size = len(lexicon)
    idx_src = list(lexicon.keys())
    similarities = compute_csls_scores(x_src, x_tgt, idx_src, k=k, bsz=bsz)
    nn = np.argpartition(-similarities, range(acc_at), axis=1)[:, :acc_at]
    correct = 0.0
    for k in range(0, len(lexicon)):
        # if nn[k] in lexicon[idx_src[k]]:
        if any(int(w) in lexicon[idx_src[k]] for w in nn[k]):
            correct += 1.0
    print(correct, len(lexicon), lexicon_size)
    return correct / lexicon_size
