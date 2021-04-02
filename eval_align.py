# -*- coding: utf-8 -*-
#
# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import io
import cupy as np
import argparse
from eval_utils import *

parser = argparse.ArgumentParser(description='Evaluation of word alignment')
parser.add_argument("--src_emb", type=str, default='', help="Load source embeddings")
parser.add_argument("--tgt_emb", type=str, default='', help="Load target embeddings")
parser.add_argument('--center', action='store_true', help='whether to center embeddings or not')
parser.add_argument("--src_mat", type=str, default='', help="Load source alignment matrix. If none given, the aligment matrix is the identity.")
parser.add_argument("--tgt_mat", type=str, default='', help="Load target alignment matrix. If none given, the aligment matrix is the identity.")
parser.add_argument("--dico_test", type=str, default='', help="test dictionary")
parser.add_argument("--maxload", type=int, default=200000)
parser.add_argument("--nomatch", action='store_true', help="no exact match in lexicon")
parser.add_argument("--report_file", type=str, help="File to write report to")
params = parser.parse_args()


###### SPECIFIC FUNCTIONS ######
# function specific to evaluation
# the rest of the functions are in utils.py

def load_transform(fname, d1=300, d2=300):
    fin = io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
    R = np.zeros([d1, d2])
    for i, line in enumerate(fin):
        tokens = line.split(' ')
        R[i, :] = np.array(tokens[0:d2], dtype=float)
    return R


###### MAIN ######

print("Evaluation of alignment on %s" % params.dico_test)
if params.nomatch:
    print("running without exact string matches")

words_tgt, x_tgt = load_vectors(params.tgt_emb, maxload=params.maxload, center=params.center)
words_src, x_src = load_vectors(params.src_emb, maxload=params.maxload, center=params.center)

if params.tgt_mat != "":
    R_tgt = load_transform(params.tgt_mat)
    x_tgt = np.dot(x_tgt, R_tgt)
if params.src_mat != "":
    R_src = load_transform(params.src_mat)
    x_src = np.dot(x_src, R_src)

src2tgt, lexicon_size = load_lexicon(params.dico_test, words_src, words_tgt)

# bsz = len(x_tgt)
bsz = 1024
k = 10
nnacc_1 = compute_nn_accuracy(x_src, x_tgt, src2tgt, lexicon_size=-1, bsz=bsz)
nnacc_3 = compute_nn_accuracy(x_src, x_tgt, src2tgt, lexicon_size=-1, bsz=bsz, acc_at=3)
nnacc_5 = compute_nn_accuracy(x_src, x_tgt, src2tgt, lexicon_size=-1, bsz=bsz, acc_at=5)
cslsproc_1 = compute_csls_accuracy(x_src, x_tgt, src2tgt, lexicon_size=-1, bsz=bsz, k=k)
cslsproc_3 = compute_csls_accuracy(x_src, x_tgt, src2tgt, lexicon_size=-1, bsz=bsz, acc_at=3, k=k)
cslsproc_5 = compute_csls_accuracy(x_src, x_tgt, src2tgt, lexicon_size=-1, bsz=bsz, acc_at=5, k=k)
print("NN@1 = %.4f - NN@3 = %.4f - NN@5 = %.4f" % (nnacc_1, nnacc_3, nnacc_5))
print("CSLS@1 = %.4f - CSLS@3 = %.4f - CSLS@5 = %.4f" % (cslsproc_1, cslsproc_3, cslsproc_5))
print("Coverage = %.4f" % (len(src2tgt) / lexicon_size))

if params.report_file:
    with open(params.report_file, "w") as f:
        f.write("P@1 = %.4f - P@3 = %.4f - P@5 = %.4f\n" % (cslsproc_1, cslsproc_3, cslsproc_5))
        f.write("P@3 = %.2f\n" % (cslsproc_3 * 100))
        f.write("Full P@3 = %.2f\n" % (cslsproc_3 * 100 * (len(src2tgt) / lexicon_size)))
