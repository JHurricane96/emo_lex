import io

def load_trans_file(filename):
    fin = io.open(filename, "r", encoding="utf-8")
    trans = []
    src_word_set = set()
    tgt_word_set = set()
    for line in fin:
        line = line.strip().split("\t")
        word_src = line[0].strip()
        tgt_words = line[1].strip().split(",")
        scores = map(float, line[2].strip().split(","))

        trans.append([word_src, list(zip(tgt_words, scores))])
        src_word_set.add(word_src)
        tgt_word_set.update(tgt_words)
    fin.close()
    return trans, src_word_set, tgt_word_set

def get_emo_ratings(trans, emo_lex_tgt):
    derived_emo_lex = {}
    for word_src, tgt_words in trans:
        emos = []
        for tgt_word, score in tgt_words:
            emo = emo_lex_tgt.get(tgt_word, None)
            if emo:
                emos.append(emo)
        if len(emos) == 0:
            continue
        derived_emo_lex[word_src] = sum(emos) / len(emos)
    return derived_emo_lex
