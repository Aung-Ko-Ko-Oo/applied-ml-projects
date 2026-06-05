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