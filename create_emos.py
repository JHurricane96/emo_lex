import io
import argparse
from collections import defaultdict
from create_emos_utils import load_trans_file, get_emo_ratings

parser = argparse.ArgumentParser(description='Creation of derived emotion lexicon')
parser.add_argument("--trans_file", type=str, default='', help="Load translations file")
parser.add_argument("--emo_lex", type=str, default='', help="Load emotion lexicon")
parser.add_argument("--induct_emos_file", type=str, default='', help="File to write induced emotions to")
params = parser.parse_args()

def load_emo_lex(emo_lex_file, words):
    fin = io.open(emo_lex_file, "r", encoding="utf-8")
    fin.readline()
    emo_lex = defaultdict(dict)
    for line in fin:
        word, emotion, rating = line.split("\t")
        rating = float(rating)
        if word in words:
            emo_lex[emotion][word] = rating
    fin.close()
    return emo_lex

def create_emo_lex(derived_emo_lex, trans, induct_emos_file, emotion):
    print("Number of derived emotion ratings:", len(derived_emo_lex))
    trans = {word_src: tgt_words for word_src, tgt_words in trans}
    for word, emo in derived_emo_lex.items():
        translations = ",".join([t[0] for t in trans[word]])
        induct_emos_file.write(f"{word}\t{translations}\t{emotion}\t{emo}\n")

if __name__ == "__main__":
    print("Creation of emotion ratings using %s" % params.trans_file)

    translations, src_words, tgt_words = load_trans_file(params.trans_file)
    emo_lex_tgt = load_emo_lex(params.emo_lex, tgt_words)
    report = []

    with open(params.induct_emos_file, "w") as induct_emos_file:
        for emotion in emo_lex_tgt.keys():
            print("\nStats for emotion:", emotion)
            single_emo_lex_tgt = emo_lex_tgt[emotion]
            derived_emo_lex_src = get_emo_ratings(translations, single_emo_lex_tgt)
            create_emo_lex(derived_emo_lex_src, translations, induct_emos_file, emotion)
