# Applied Machine Learning Projects

A curated collection of end-to-end Machine Learning, Data Science, and Natural Language Processing (NLP) projects showcasing practical implementations, dataset preprocessing, models, and benchmark analyses.

---

## Repository Overview

| Project Directory | Category | Domain / Focus | Key Technologies |
| :--- | :--- | :--- | :--- |
| **`1-california-housing`** | Tabular / Regression | Housing Price Prediction & Feature Engineering | Python, Pandas, Scikit-Learn, XGBoost |
| **`2-nasa-pace-ocean`** | Remote Sensing / Spatial | Ocean Color & Earth Science Data Analysis | Python, NetCDF4, Xarray, Rasterio, PyTorch |
| **`3-POLAR-benchmark-myanmar`** | NLP / Classification | Sentiment, Polarization & Toxicity Analysis | Python, Hugging Face `datasets`, Pandas, Regex |
| **`4-myanmar-hieroglyph-nmt`** | NLP / Translation | Myanmar ↔ Hieroglyph Neural Machine Translation | Python, Marian NMT, SentencePiece, sacreBLEU |

---

## Project Summaries

### 1. California Housing Price Prediction (`1-california-housing`)
* **Objective**: Predict median house values in California districts using geographical, demographic, and structural metrics.
* **Key Tasks**: Exploratory Data Analysis (EDA), missing value imputation, spatial coordinate analysis, feature scaling, and evaluation using RMSE/MAE metrics across multiple regression models.

### 2. NASA PACE Ocean Color Data Analysis (`2-nasa-pace-ocean`)
* **Objective**: Process and analyze satellite remote sensing data from NASA's Plankton, Aerosol, Cloud, ocean Ecosystem (PACE) mission.
* **Key Tasks**: Multi-spectral imagery extraction, ocean color parameter estimation (chlorophyll-a concentration), spatial grid mapping, and atmospheric correction modeling.

### 3. POLAR Benchmark Dataset: Myanmar (`3-POLAR Benchmark Dataset...`)
* **Objective**: Build a 600-sentence benchmark dataset for Myanmar (Burmese) language sentiment, polarization, and toxicity classification adhering to POLAR guidelines.
* **Key Tasks**: Wikipedia web scraping, Hugging Face dataset sampling (`simbolo-ai/burmese-hatespeech`), custom Burmese syllable/word segmentation, and multi-label annotation (Toxicity, Insult, Identity Attack).

### 4. Myanmar ↔ Hieroglyph Neural Machine Translation (`4-myanmar-hieroglyph-nmt`)
* **Objective**: Build a Neural Machine Translation (NMT) system for bidirectional translation between **Myanmar** and **Hieroglyph** using the **Marian NMT** framework, comparing **Seq2Seq**, **Transformer**, and **Ensemble** architectures.
* **Key Tasks**: Data visualization and analysis, SentencePiece subword tokenization (8k vocab per language), training 4 models (2 architectures × 2 directions), ensemble decoding, BLEU evaluation, and error analysis.
* **Results**: Ensemble decoding achieves **BLEU 19.54** (My→Hi) and **BLEU 21.55** (Hi→My) — outperforming both individual architectures.
* **Dataset**: 8,500 train / 1,000 valid / 500 test sentences (source: [AIE-F-B2](https://github.com/ye-kyaw-thu/AIE-F-B2))
* **Models**: [Download from Releases](https://github.com/Aung-Ko-Ko-Oo/applied-ml-projects/releases/tag/v1.0.0) (~230 MB ZIP)

---

## Directory Structure

```text
applied-ml-projects/
│
├── 1-california-housing/
│   ├── data/                   # Dataset files
│   ├── notebooks/              # Exploratory analysis & modeling
│   └── README.md               # Project-specific documentation
│
├── 2-nasa-pace-ocean/
│   ├── data/                   # Satellite NetCDF/HDF5 data
│   ├── notebooks/              # Spatial processing & visual models
│   └── README.md               # Project-specific documentation
│
├── 3-POLAR Benchmark Dataset: Myanmar Sentiment & Toxicity Analysis/
│   ├── POLAR_Final_Submission/ # Final exported task CSV files
│   ├── mya_formal_sentences_300.csv
│   ├── mya_social_simbolo_300.csv
│   └── README.md               # Project & dataset documentation
│
├── 4-myanmar-hieroglyph-nmt/
│   ├── configs/                # Marian YAML configs
│   ├── vocab/                  # SentencePiece models (8k vocab)
│   ├── data/                   # Tokenized train/valid/test data
│   ├── results/                # BLEU scores & prediction files
│   ├── docs/                   # Training logs
│   ├── models/                 # Model checkpoints (download separately)
│   └── README.md               # Project-specific documentation
│
├── .gitignore
└── README.md                   # Main repository overview
```

---

## 🏆 Highlighted Results — Project 4

| Direction | Architecture | BLEU |
|-----------|-------------|------|
| Myanmar → Hieroglyph | Transformer | 2.95 |
| Myanmar → Hieroglyph | Seq2Seq | 14.31 |
| **Myanmar → Hieroglyph** | **Ensemble** | **19.54** ⭐ |
| Hieroglyph → Myanmar | Transformer | 3.62 |
| Hieroglyph → Myanmar | Seq2Seq | 13.58 |
| **Hieroglyph → Myanmar** | **Ensemble** | **21.55** ⭐ |

**Download models**: [Releases v1.0.0](https://github.com/Aung-Ko-Ko-Oo/applied-ml-projects/releases/tag/v1.0.0)

---

## 📜 License

MIT License — see [LICENSE](LICENSE).

---

## 🙏 Acknowledgements

- **Project 3 dataset**: [simbolo-ai/burmese-hatespeech](https://huggingface.co/datasets/simbolo-ai/burmese-hatespeech) on HuggingFace
- **Project 4 dataset**: [AIE-F-B2](https://github.com/ye-kyaw-thu/AIE-F-B2) by Prof Dr.Ye Kyaw Thu
- **Project 4 framework**: [Marian NMT](https://marian-nmt.github.io/), [SentencePiece](https://github.com/google/sentencepiece)
