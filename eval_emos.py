import io
import cupy as np
import argparse
from collections import defaultdict
from csv_helpers import write_all_rows
from eval_utils import *
from create_emos_utils import load_trans_file, get_emo_ratings

parser = argparse.ArgumentParser(description='Evaluation of emotions')
parser.add_argument("--trans_file", type=str, default='', help="Load translations file")
parser.add_argument("--emo_lex", type=str, default='', help="Load emotion lexicon")
parser.add_argument("--report_file", type=str, default='', help="File to write report to")
parser.add_argument("--induct_emos_file", type=str, default='', help="File to write induced emotions to")
parser.add_argument("--induct_emos_eval_file", type=str, default='', help="File to write evaluation of induced emotions to")
params = parser.parse_args()

def load_emo_lex(emo_lex_file, src_words, tgt_words):
    fin = io.open(emo_lex_file, "r", encoding="utf-8")
    fin.readline()
    emo_lex_src = defaultdict(dict)
    emo_lex_tgt = defaultdict(dict)
    src_word_count = defaultdict(lambda: defaultdict(int))
    for line in fin:
        word_tgt, word_src, emotion, rating = line.split("\t")
        rating = float(rating)
        if word_tgt in tgt_words:
            emo_lex_tgt[emotion][word_tgt] = rating
        if (word_src != "NO TRANSLATION"):
            src_word_count[emotion][word_src] += 1
            word_count = src_word_count[emotion][word_src]
            if word_count == 1:
                emo_lex_src[emotion][word_src] = rating
            else:
                emo_lex_src[emotion][word_src] = ((emo_lex_src[emotion][word_src] * (word_count - 1)) + rating)/word_count
    fin.close()
    return emo_lex_src, emo_lex_tgt

def eval_emo_lex(derived_emo_lex, emo_lex, trans, induct_emos_file, induct_emos_eval_file, emotion):
    print("Number of derived emotion ratings:", len(derived_emo_lex))
    derived_emos = []
    real_emos = []
    words = []
    trans = {word_src: tgt_words for word_src, tgt_words in trans}
    for word, emo in derived_emo_lex.items():
        translations = ",".join([t[0] for t in trans[word]])
        induct_emos_file.write(f"{word}\t{translations}\t{emotion}\t{emo}\n")
        real_emo = emo_lex.get(word, None)
        if real_emo:
            induct_emos_eval_file.write(f"{word}\t{translations}\t{emotion}\t{emo}\t{real_emo}\n")
            derived_emos.append(emo)
            real_emos.append(real_emo)
            words.append(word)
    
    print("Coverage in test set:", len(derived_emos) / len(derived_emo_lex))

    derived_emos = np.array(derived_emos, dtype=float)
    real_emos = np.array(real_emos, dtype=float)
    corr_coeff = np.corrcoef(derived_emos, real_emos, rowvar=False)
    top_words = np.argsort(-derived_emos)[:10]
    print(derived_emos[top_words])
    top_words = [words[int(idx)] for idx in top_words]
    print(top_words)
    corr_coeff = np.around(corr_coeff[0, 1], 3)
    print("Correlation:", corr_coeff)
    return [corr_coeff, len(derived_emo_lex), derived_emos.shape[0]]

if __name__ == "__main__":
    print("Evaluation of emotion ratings on %s" % params.trans_file)

    translations, src_words, tgt_words = load_trans_file(params.trans_file)
    emo_lex_src, emo_lex_tgt = load_emo_lex(params.emo_lex, src_words, tgt_words)
    report = []

    with open(params.induct_emos_file, "w") as induct_emos_file,\
        open(params.induct_emos_eval_file, "w") as induct_emos_eval_file:
        for emotion in emo_lex_src.keys():
            print("\nStats for emotion:", emotion)
            single_emo_lex_src = emo_lex_src[emotion]
            single_emo_lex_tgt = emo_lex_tgt[emotion]
            derived_emo_lex_src = get_emo_ratings(translations, single_emo_lex_tgt)
            report_record = eval_emo_lex(derived_emo_lex_src, single_emo_lex_src, translations, induct_emos_file, induct_emos_eval_file, emotion)
            report_record.insert(0, emotion)
            report.append(report_record)

    write_all_rows(params.report_file, report)
