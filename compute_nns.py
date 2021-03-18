import io
import argparse
from eval_utils import *

parser = argparse.ArgumentParser(description='Computation of nearest neighbors')
parser.add_argument("--src_emb", type=str, default='', help="Load source embeddings")
parser.add_argument("--tgt_emb", type=str, default='', help="Load target embeddings")
parser.add_argument('--center', action='store_true', help='whether to center embeddings or not')
parser.add_argument("--dico_test", type=str, default='', help="test dictionary")
parser.add_argument("--nn_words", type=str, default='', help="Words to get nearest neighbours for")
parser.add_argument("--maxload", type=int, default=200000)
parser.add_argument("--nns_file", type=str, default='', help="Path to save nearest neighbours")
params = parser.parse_args()

def save_nns(filename, nns, words_src, words_tgt):
    fout = io.open(filename, "w", encoding="utf-8")
    for src_idx, (tgt_idx, scores, correct) in nns.items():
        tgt_words = ",".join((words_tgt[int(idx)] for idx in tgt_idx))
        fout.write("%s\t%s\t%s\t%d\n" % (words_src[int(src_idx)], tgt_words, ",".join(map(lambda s: str(s.round(4)), scores)), int(correct)))
    fout.close()

def load_nn_words(filename):
    fin = io.open(filename, "r", encoding="utf-8")
    nn_words = {line.strip() for line in fin}
    fin.close()
    return nn_words

print("Computing nearest neighbours based on %s" % params.dico_test)

words_tgt, x_tgt = load_vectors(params.tgt_emb, maxload=params.maxload, center=params.center)
words_src, x_src = load_vectors(params.src_emb, maxload=params.maxload, center=params.center)

nn_words = set()
if params.nn_words:
    nn_words = load_nn_words(params.nn_words)

src2tgt = None
if params.dico_test != '':
    src2tgt, lexicon_size = load_lexicon(params.dico_test, words_src, words_tgt)
nns = compute_csls_maps(x_src, words_src, x_tgt, src2tgt, nn_words, 3, k=10)
save_nns(params.nns_file, nns, words_src, words_tgt)
