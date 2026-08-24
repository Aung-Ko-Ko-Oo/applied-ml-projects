### 2. [NASA PACE Satellite Ocean Biomass Predictor](./2-nasa-pace-ocean)
* **Domain:** Climate Science / Remote Sensing / Geospatial Regression
* **Objective:** Forecast global marine Chlorophyll-a density concentrations using satellite telemetry streams from the NASA PACE mission.
* **Core Tech:** Gradient Boosting Regressors (GBM), Continuous Spatial Normalization, Feature Pipeline Engineering.

---


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
