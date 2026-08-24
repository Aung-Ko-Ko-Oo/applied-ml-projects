 Machine Learning & Remote Sensing Portfolio

Welcome to my professional machine learning portfolio. This repository serves as a centralized hub for production-grade, end-to-end ML pipelines, spanning classical economic regressions to advanced geospatial climate analytics.

---

## Repository Projects

### 1. [California Housing Valuation Engine](./1-california-housing)
* **Domain:** Real Estate Economics / Tabular Regression
* **Objective:** Forecast regional median residential real estate valuations across California block groups.
* **Core Tech:** Python, Scikit-Learn, Random Forest Regressor, Robust Preprocessing Pipelines.


# Project 1: California Housing Valuation Engine

## Pipeline Execution Instructions
From this directory, execute the pipeline step-by-step:

```bash
# Install dependencies
pip install -r requirements.txt
pip install -e .

# Run stages sequentially
python src/ingest_data.py
python src/preprocess.py
python src/train.py

# Test inference execution
python src/predict.py
