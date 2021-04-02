# emo_lex

Experiment runner for paper [Cross-Lingual Emotion Lexicon Induction using Representation Alignment in Low-Resource Settings](https://www.aclweb.org/anthology/2020.coling-main.517/).

# Usage

### Setup

1. Clone this repo
2. Clone [fastText](https://github.com/JHurricane96/fastText), [multilingual-nlm](https://github.com/JHurricane96/multilingual-nlm) and [vecmap](https://github.com/JHurricane96/vecmap).
3. Install python packages `numpy`, `cupy`, `torch` and <code>[pot](https://pot.readthedocs.io/en/stable/)</code>

## Embedding Alignment

Some conventions:

- Languages are referred to by their 3 letter ISO code.
- Each bible file should be named `<iso>.txt`, e.g. Spanish would be `spa.txt`. 
- Each line in a Bible file should contain 1 sentence/verse. The text should be pre-processed to be lowercased and contain space-separated words (no punctuation unless it's in the middle of a word, e.g. hypenation).

The 3 algorithms used for embedding alignment:

### Wasserstein-Procrustes

```
python align.py \
  --langs <space-separated list of languages> \
  --bible_dir <directory with Bibles> \
  --align_dir <directory to save aligned embeddings> \
  --emb_dir <directory to save initial unaligned fasttext embeddings> \
  --num_gpus <number of GPUs to use> \
  --algorithm fb \
  --fasttext_dir <path to fastText clone>
```

### Neural Language Model

```
python align.py \
  --langs <space-separated list of languages> \
  --bible_dir <directory with Bibles> \
  --align_dir <directory to save aligned embeddings> \
  --num_gpus <number of GPUs to use> \
  --algorithm nlm \
  --nlm_dir <path to nlm clone> \
  --nlm_preproc_dir <directory to save preprocessed stuff like vocabulary>  \
  --nlm_preprocess <flag to not skip preprocessing, can remove after running once> \
  --nlm_modified <run modified version, not original>
```

### Orthogonal Refinement

```
python align.py \
  --langs <space-separated list of languages> \
  --bible_dir <directory with Bibles> \
  --sid_bible_dir <directory with Bibles with first column having sentence ID> \
  --align_dir <directory to save aligned embeddings> \
  --emb_dir <directory to save initial unaligned fasttext embeddings> \
  --num_gpus <number of GPUs to use> \
  --algorithm vecmap \
  --vecmap_dir <path to vecmap clone> \
  --fasttext_dir <path to fastText clone>
```

The sentence ID Bibles are like the normal Bibles, except each line is prefixed with a sentence ID followed by a tab. Sentence ID for translations of the same sentence across different language should be the same.

Omit the `--sid_bible_dir` argument to run the original vecmap algorithm.

## Emotion Lexicon Induction and Evaluation

```
python eval.py \
  --langs <space-separated list of languages> \
  --align_dir <directory with aligned embeddings from previous step> \
  --exp_id <experiment ID> \
  --trans_dir <directory with ground-truth word translations> \
  --emo_lex_dir <directory with ground-truth emotion lexicons> \
  --nns_dir <directory to save derived word translations to> \
  --reports_dir <directory to save evaluation reports and induced emotion lexicons>
```

- The path provided to `nns_dir` and `reports_dir` is suffixed by `exp_id` so that multiple runs can share the same paths and just have different experiment IDs.
- The ground-truth word translation files should be named `<src_iso>_<tgt_iso>.txt`, e.g. `spa_eng.txt` for Spanish-to-English translations. They should be tab-separated files, with the first column being a word in the source language and the second column being the translation of the word into the target language.
- The ground-truth emotion lexicons should be named `<iso>.txt`. They can be obtained from the [NRC EIL webpage](https://saifmohammad.com/WebPages/AffectIntensity.htm).

To run only the lexicon induction and skip the evaluation:

```
python eval.py \
  --langs <space-separated list of languages> \
  --align_dir <directory with aligned embeddings from previous step> \
  --exp_id <experiment ID> \
  --nns_dir <directory to save derived word translations to> \
  --reports_dir <directory to save evaluation reports and induced emotion lexicons> \
  --skip_eval
```