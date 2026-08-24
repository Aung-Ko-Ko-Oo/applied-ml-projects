# POLAR Benchmark Dataset: Myanmar Sentiment & Toxicity Analysis

A benchmark dataset and automated processing pipeline for Myanmar (Burmese) language sentiment, polarization, and toxicity classification adhering to the **POLAR Benchmark Specification**.

---

## Overview

This repository contains data scraping, word segmentation, and automated labeling scripts along with standardized datasets designed for NLP tasks in the Myanmar language. The dataset combines curated formal text (literature/Wikipedia) and user-generated social media content (Hugging Face) to evaluate sentiment, hate speech, and toxicity.

---

## Dataset Summary

| Metric / Parameter | Value | Details |
| :--- | :--- | :--- |
| **Total Sentences** | 600 | Split equally across formal and social media domains |
| **Part 1 (Formal)** | 300 sentences | Wikipedia articles covering historical & diplomatic topics (`mya_formal_sentences_300.csv`) |
| **Part 2 (Social Media)** | 300 sentences | Randomly sampled from `simbolo-ai/burmese-hatespeech` (`mya_social_simbolo_300.csv`) |
| **Language** | Myanmar (Burmese) | Encoded in UTF-8 (`utf-8-sig`) |
| **ID Format** | `mya_XXXXXXXX` | Unique 8-character hexadecimal identifier |

---

## Directory Structure

```text
├── mya_formal_sentences_300.csv       # Scraped formal Wikipedia sentences (300 rows)
├── mya_social_simbolo_300.csv        # Processed social media comments (300 rows)
├── POLAR_Final_Submission/
│   ├── polar_task1_raw_scraped.csv    # Merged raw text data with metadata (600 rows)
│   ├── polar_task2_word_segmented.csv # Syllable/word-segmented Burmese text (600 rows)
│   └── polar_task3_fully_labelled.csv  # Final dataset with full POLAR annotations (600 rows)
└── README.md                          # Project documentation
## Quick Start & Running the Pipeline

### Prerequisites
Install the required dependencies using `pip`:

```bash
pip install pandas datasets

import pandas as pd

# Load Task 3 output dataset
df_task3 = pd.read_csv("POLAR_Final_Submission/polar_task3_fully_labelled.csv")

# Filter toxic comments with keyphrases
toxic_samples = df_task3[df_task3["Toxicity"] == 1][["segmented_text", "Keyphrase", "Source Link"]]

print(f"Total Toxic Sentences Extracted: {len(toxic_samples)}")
print(toxic_samples.head())

## Citation & Citation Formats

If you use this benchmark dataset, code pipeline, or the underlying `simbolo-ai/burmese-hatespeech` dataset in your research or project, please cite this repository and the original sources using the formats below:

### BibTeX

```bibtex
@dataset{polar_myanmar_2026,
  author       = {POLAR Benchmark Team},
  title        = {POLAR Benchmark Dataset: Myanmar Sentiment, Polarization & Toxicity Analysis},
  year         = {2026},
  publisher    = {AUNG KO KO OO},
  journal      = {https://github.com/Aung-Ko-Ko-Oo/applied-ml-projects},
  howpublished = {\url{[https://github.com/Aung-Ko-Ko-Oo/polar-myanmar-benchmark](https://github.com/your-username/polar-myanmar-benchmark)}}
}

@dataset{simbolo_burmese_hatespeech,
  author       = {Simbolo AI},
  title        = {Burmese Hatespeech Dataset},
  year         = {2023},
  publisher    = {Hugging Face},
  howpublished = {\url{[https://huggingface.co/datasets/simbolo-ai/burmese-hatespeech](https://huggingface.co/datasets/simbolo-ai/burmese-hatespeech)}}
}
