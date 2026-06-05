# Project 2: NASA PACE Satellite Ocean Biomass Predictor

## Pipeline Execution Instructions
From this directory, execute the pipeline step-by-step:

```bash
# Install dependencies
pip install -r requirements.txt
pip install -e .

# Run stages sequentially
python src/ingest_pace_data.py
python src/transform_pipeline.py
python src/train_evaluate.py

# Test inference execution
python src/serve_inference.py