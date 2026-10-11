# Applied Machine Learning Projects

A curated collection of end-to-end Machine Learning, Data Science, and Natural Language Processing (NLP) projects showcasing practical implementations, dataset preprocessing, models, and benchmark analyses.

---

## Repository Overview

| Project Directory | Category | Domain / Focus | Key Technologies |
| :--- | :--- | :--- | :--- |
| **`1-california-housing`** | Tabular / Regression | Housing Price Prediction & Feature Engineering | Python, Pandas, Scikit-Learn, XGBoost |
| **`2-nasa-pace-ocean`** | Remote Sensing / Spatial | Ocean Color & Earth Science Data Analysis | Python, NetCDF4, Xarray, Rasterio, PyTorch |
| **`3-POLAR-benchmark-myanmar`** | NLP / Classification | Sentiment, Polarization & Toxicity Analysis | Python, Hugging Face `datasets`, Pandas, Regex |

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
├── .gitignore
└── README.md                   # Main repository overview
