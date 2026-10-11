# Myanmar ↔ Hieroglyph Neural Machine Translation

Neural Machine Translation (NMT) system for **Myanmar ↔ Hieroglyph** translation using the **Marian NMT** framework. This project compares **Seq2Seq (RNN)**, **Transformer**, and **Ensemble** decoding on a parallel corpus of 10,000 sentence pairs.

> **Course**: AIE-F-B2 — Assignment 6  
> **Student**: Aung Ko Ko Oo  
> **Instructor**: Prof. Dr. Ye Kyaw Thu  
> **Date**: October 2026

---

## 📊 Results

| Direction | Architecture | BLEU |
|-----------|-------------|------|
| Myanmar → Hieroglyph | Transformer | 2.95 |
| Myanmar → Hieroglyph | Seq2Seq | 14.31 |
| **Myanmar → Hieroglyph** | **Ensemble** | **19.54** ⭐ |
| Hieroglyph → Myanmar | Transformer | 3.62 |
| Hieroglyph → Myanmar | Seq2Seq | 13.58 |
| **Hieroglyph → Myanmar** | **Ensemble** | **21.55** ⭐ |

### Key Findings

1. **Ensemble decoding wins consistently** — combining Seq2Seq + Transformer gives **+5.2 BLEU** (My→Hi) and **+7.9 BLEU** (Hi→My) over the best single model. The two architectures make uncorrelated errors, so averaging their softmax probabilities reduces variance.

2. **Seq2Seq ≫ Transformer on small data** — With only 8,500 training sentences, Seq2Seq achieves **~4.7× higher BLEU** than Transformer. Transformer requires much more data to learn attention patterns effectively.

3. **Direction asymmetry** — Hi→My slightly outperforms My→Hi across all models (+2.0 BLEU on Ensemble). Myanmar output has richer vocabulary which the decoder handles well.

Full results: [`results/final_results.csv`](results/final_results.csv)

---

## 📁 Dataset

| Split | Sentences | Avg. Chars (Myanmar) | Avg. Chars (Hieroglyph) |
|-------|-----------|---------------------|------------------------|
| Train | 8,500 | 20.2 | 13.5 |
| Valid | 1,000 | 19.7 | 13.6 |
| Test  | 500  | 20.4 | 13.4 |

- **Myanmar**: 109 unique characters, 7,049 unique words, 163,579 total chars
- **Hieroglyph**: 2,858 unique tokens (Egyptian Unicode, U+13000–U+1342F), 26,835 total tokens

**Source**: [AIE-F-B2/assignment/assignment-6](https://github.com/ye-kyaw-thu/AIE-F-B2) by Ye Kyaw Thu

### Sample Pairs

| Myanmar | Hieroglyph |
|---------|-----------|
| မင်းက အေးဆေးတယ် | 𓂨𓀀𓎡𓋴𓂝 𓏮𓀀𓐍𓂋 |
| ရိုးသားမှုရှိတယ် | 𓏏𓂝𓋴𓂝 𓈙𓆑𓎛𓏭𓂝𓋴𓂝 |
| ဘယ်သူမှ ကျွန်မကို မမြင်ခဲ့ဘူး။ | 𓎜𓀀 𓐍𓎛𓂝 𓈙𓄿𓏏 𓉔𓂝 |

---

## 🏗️ Architecture

```
             Myanmar / Hieroglyph Input
                       │
                       ▼
        ┌─────────────────────────────┐
        │  SentencePiece Tokenizer    │
        │  (Unigram, 8,000 vocab)     │
        └─────────────────────────────┘
                       │
          ┌────────────┴────────────┐
          ▼                         ▼
   ┌─────────────┐          ┌──────────────┐
   │  Seq2Seq    │          │  Transformer │
   │  dim-emb:256│          │  dim-emb:256 │
   │  dim-rnn:512│          │  enc/dec: 3  │
   │  enc/dec: 2 │          │  heads: 4    │
   └─────────────┘          └──────────────┘
          │                         │
          └────────────┬────────────┘
                       ▼
              ┌────────────────┐
              │    Ensemble    │
              │  Beam size: 4  │
              └────────────────┘
                       │
                       ▼
             Hieroglyph / Myanmar Output
```

---

## 📂 Repository Structure

```
4-myanmar-hieroglyph-nmt/
├── README.md                    ← This file
├── requirements.txt             ← Python dependencies
├── configs/                     ← Marian YAML configs
│   ├── config_transformer.yml          (My → Hi, Transformer)
│   ├── config_transformer_reverse.yml  (Hi → My, Transformer)
│   ├── config_s2s.yml                  (My → Hi, Seq2Seq)
│   ├── config_s2s_reverse.yml          (Hi → My, Seq2Seq)
│   ├── ensemble_my_hi.decoder.yml      (My → Hi, Ensemble)
│   └── ensemble_hi_my.decoder.yml      (Hi → My, Ensemble)
├── vocab/                       ← SentencePiece models
│   ├── my_spm.model, my_spm.vocab, my_spm.marian.vocab
│   └── hi_spm.model, hi_spm.vocab, hi_spm.marian.vocab
├── data/                        ← Encoded data (see data/README.md)
│   ├── train.my.sp, train.hi.sp
│   ├── valid.my.sp, valid.hi.sp
│   └── test.my.sp, test.hi.sp
├── results/                     ← Predictions & BLEU scores
│   ├── final_results.csv
│   ├── test.hi.pred.transformer.text
│   ├── test.hi.pred.s2s.text
│   ├── test.hi.pred.ensemble.text
│   ├── test.my.pred.transformer.text
│   ├── test.my.pred.s2s.text
│   └── test.my.pred.ensemble.text
├── docs/                        ← Training logs
│   └── train_tf_my_hi.log
└── models/                      ← Model checkpoints (see models/README.md)
    └── README.md                ← Download link (GitHub Releases)
```

---

## 🚀 Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/Aung-Ko-Ko-Oo/applied-ml-projects.git
cd applied-ml-projects/4-myanmar-hieroglyph-nmt
```

### 2. Install Python dependencies

```bash
pip install -r requirements.txt
```

### 3. Compile Marian NMT

Marian NMT is a C++ framework — install it from source:

```bash
git clone https://github.com/marian-nmt/marian.git
cd marian
git submodule update --init --recursive
mkdir build && cd build
cmake .. \
    -DCOMPILE_CPU=on \
    -DCOMPILE_CUDA=off \
    -DUSE_SENTENCEPIECE=on \
    -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

This takes ~5–10 minutes on CPU. The binaries will be in `marian/build/`:
- `marian` — training
- `marian-decoder` — inference

### 4. Download model checkpoints

Model weights are hosted on **[GitHub Releases](https://github.com/Aung-Ko-Ko-Oo/applied-ml-projects/releases/tag/v1.0.0)** (each file is 46–67 MB, exceeding in-repo limits):

```bash
wget https://github.com/Aung-Ko-Ko-Oo/applied-ml-projects/releases/download/v1.0/models.zip
unzip models.zip -d models/
```

Expected files in `models/`:

| File | Size | Architecture |
|------|------|--------------|
| `model_transformer_my_hi.npz` | 46.3 MB | Transformer My→Hi |
| `model_s2s_my_hi.npz` | 64.3 MB | Seq2Seq My→Hi |
| `model_transformer_hi_my.npz` | 49.7 MB | Transformer Hi→My |
| `model_s2s_hi_my.npz` | 67.7 MB | Seq2Seq Hi→My |

### 5. Translate

```bash
MARIAN_DECODER=/path/to/marian/build/marian-decoder

# Myanmar → Hieroglyph (Ensemble — best result)
$MARIAN_DECODER \
    -c configs/ensemble_my_hi.decoder.yml \
    --input data/test.my.sp \
    --output results/my_output.txt

# Hieroglyph → Myanmar (Ensemble — best result)
$MARIAN_DECODER \
    -c configs/ensemble_hi_my.decoder.yml \
    --input data/test.hi.sp \
    --output results/my_output_reverse.txt
```

---

## 🏋️ Training

### Config Files

| Config | Direction | Architecture |
|--------|-----------|--------------|
| `configs/config_transformer.yml` | My → Hi | Transformer |
| `configs/config_s2s.yml` | My → Hi | Seq2Seq |
| `configs/config_transformer_reverse.yml` | Hi → My | Transformer |
| `configs/config_s2s_reverse.yml` | Hi → My | Seq2Seq |

### Train Transformer (Myanmar → Hieroglyph)

```bash
marian -c configs/config_transformer.yml --cpu-threads 4
# Training time: ~70 minutes (CPU, 4 threads)
```

### Train Seq2Seq (Myanmar → Hieroglyph)

```bash
marian -c configs/config_s2s.yml --cpu-threads 4
# Training time: ~172 minutes (CPU, 4 threads)
```

### Hyperparameters

**Transformer**:
```yaml
type: transformer
dim-emb: 256
enc-depth: 3
dec-depth: 3
transformer-heads: 4
transformer-dropout: 0.1
label-smoothing: 0.1
learn-rate: 0.0003
lr-warmup: 2000
lr-decay-inv-sqrt: 2000
clip-norm: 5
optimizer: adam
optimizer-params: [0.9, 0.98, 1e-09]
mini-batch: 32
beam-size: 4
```

**Seq2Seq**:
```yaml
type: s2s
dim-emb: 256
dim-rnn: 512
enc-depth: 2
dec-depth: 2
dropout-rnn: 0.2
dropout-src: 0.1
dropout-trg: 0.1
learn-rate: 0.0003
lr-warmup: 2000
clip-norm: 5
optimizer: adam
optimizer-params: [0.9, 0.98, 1e-09]
mini-batch: 32
beam-size: 4
```

**Common**:
- `early-stopping: 5` (stop after 5 validations without improvement)
- `valid-freq: 500`, `save-freq: 500`
- `workspace: 2000` MB

---

## 🔤 Tokenization (SentencePiece)

Both languages use **SentencePiece unigram** with an 8,000-token vocabulary:

```python
import sentencepiece as spm

spm.SentencePieceTrainer.train(
    input='all.my',
    model_prefix='vocab/my_spm',
    vocab_size=8000,
    character_coverage=1.0,
    model_type='unigram',
    num_threads=4,
    unk_id=0, bos_id=1, eos_id=2, pad_id=3,
    unk_piece='<unk>', bos_piece='<s>', eos_piece='</s>', pad_piece='<pad>',
    hard_vocab_limit=False
)
```

**Special tokens** (required by Marian):
- `<unk>` = 0, `<s>` = 1, `</s>` = 2, `<pad>` = 3

**Encoded sample**:
```
Myanmar:    <s> ▁မင်းက ▁အေးဆေးတယ် </s>
Hieroglyph: <s> ▁𓂨𓀀𓎡𓋴𓂝 ▁𓏮𓀀𓐍𓂋 </s>
```

---

## 🧪 Ensemble Decoding

Ensemble combines the two architectures by averaging their output probability distributions:

```yaml
# configs/ensemble_my_hi.decoder.yml
relative-paths: true
models:
  - ../models/model_transformer_my_hi.npz
  - ../models/model_s2s_my_hi.npz
vocabs:
  - ../data/train.my.sp.yml
  - ../data/train.hi.sp.yml
beam-size: 4
normalize: 0.6
word-penalty: 0
mini-batch: 32
maxi-batch: 100
maxi-batch-sort: src
```

Run:
```bash
marian-decoder \
    -c configs/ensemble_my_hi.decoder.yml \
    --input data/test.my.sp \
    --output results/test.hi.pred.ensemble.txt
```

**Why it works**: The Seq2Seq and Transformer make different types of mistakes. Averaging their softmax probabilities is a form of model ensembling that reduces variance without retraining.

---

## 📈 Evaluation

Compute BLEU with [sacreBLEU](https://github.com/mjpost/sacrebleu):

```bash
sacrebleu test.hi -i results/test.hi.pred.ensemble.text -m bleu
```

Or in Python:
```python
import sacrebleu

with open('test.hi') as f:
    refs = [l.strip() for l in f if l.strip()]
with open('results/test.hi.pred.ensemble.text') as f:
    hyps = [l.strip() for l in f if l.strip()]

bleu = sacrebleu.corpus_bleu(hyps, [refs]).score
print(f"BLEU: {bleu:.2f}")
```

### Sample Translations (Ensemble, Test Set)

**Myanmar → Hieroglyph**:

| Source (Myanmar) | Predicted (Hieroglyph) | Reference |
|------------------|----------------------|-----------|
| အချစ်က နာကျင်စေတယ်။ | 𓏏𓊪𓐍 𓊪𓎡𓋴𓈎 | 𓂋𓐍𓎼𓂝 𓊪𓎡𓋴𓈎𓈙 |
| ကျွန်မမှာ အစားအစာမရှိဘူး။ | 𓆑 𓊪𓄿𓎼𓂝 𓎜𓀀 𓅲𓀀𓐍𓇌 | 𓆑 𓊪𓄿𓎼𓂝 𓎜𓀀 𓅲𓀀𓐍𓇌 |
| ဘယ်သူမှ ကျွန်မကို မမြင်ခဲ့ဘူး။ | 𓎜𓀀 𓐍𓎛𓂝 𓈙𓄿𓏏 𓉔𓂝 | 𓎜𓀀 𓐍𓎛𓂝 𓈙𓄿𓏏 𓉔𓂝 |

**Hieroglyph → Myanmar**:

| Source (Hieroglyph) | Predicted (Myanmar) | Reference |
|--------------------|---------------------|-----------|
| 𓆑 𓊪𓄿𓎼𓂝 𓎜𓀀 𓅲𓀀𓐍𓇌 | ကျွန်မမှာ အစားအစာမရှိဘူး။ | ကျွန်မမှာ အစားအစာမရှိဘူး။ |
| 𓎜𓀀 𓐍𓎛𓂝 𓈙𓄿𓏏 𓉔𓂝 | ဘယ်သူမှ ကျွန်မကို မမြင်ခဲ့ဘူး။ | ဘယ်သူမှ ကျွန်မကို မမြင်ခဲ့ဘူး။ |

---

## 📊 Error Analysis

| Model | Exact Match | Avg Length Ratio |
|-------|-------------|------------------|
| My→Hi Transformer | 1.41% | 0.96 |
| My→Hi Seq2Seq | 4.60% | 0.96 |
| Hi→My Transformer | 2.04% | 1.00 |
| Hi→My Seq2Seq | **10.80%** | 0.96 |

**Length ratios** hover around 0.96–1.00, meaning predictions are approximately the right length. Higher exact-match rates for Seq2Seq reflect its tendency to produce safer, more common outputs.

---

## 📜 License

MIT License — see [LICENSE](../../LICENSE).

The dataset is licensed under the original [AIE-F-B2](https://github.com/ye-kyaw-thu/AIE-F-B2) repository terms.

---

## 🙏 Acknowledgements

- [Marian NMT](https://marian-nmt.github.io/) — Training framework
- [SentencePiece](https://github.com/google/sentencepiece) — Subword tokenizer
- [sacreBLEU](https://github.com/mjpost/sacrebleu) — BLEU evaluation
- [AIE-F-B2](https://github.com/ye-kyaw-thu/AIE-F-B2) — Dataset (Ye Kyaw Thu)

---

## 📧 Contact

**Aung Ko Ko Oo**  
📧 AungKoKoOo@my.uopeopl.edu
🔗 [GitHub](https://github.com/Aung-Ko-Ko-Oo)
