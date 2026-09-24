# ApexPace

**End-to-end ML system for predicting Formula 1 lap times from sequential race data.**

ApexPace uses historical F1 race data to predict a driver's next lap time from recent pace and tyre-state features. The project covers the full ML workflow: data ingestion, feature engineering, temporal validation, baseline comparison, XGBoost training, hyperparameter tuning, experiment tracking, API serving, and automated testing.

## Project Overview

The model predicts:

> **Next lap time in seconds**

using:

* Previous lap time
* Three-lap rolling mean
* Tyre compound
* Fresh-tyre indicator
* Tyre life

The project uses **2023 Bahrain Grand Prix data for training/validation** and **2024 Bahrain Grand Prix data as a held-out test set**.

### Architecture

```text
FastF1
   ↓
DuckDB
   ↓
Polars feature engineering
   ↓
Temporal train / validation split
   ↓
Baseline models
   ↓
XGBoost
   ↓
Optuna hyperparameter tuning
   ↓
W&B experiment tracking
   ↓
Saved model
   ↓
FastAPI
   ↓
pytest
```

## Setup

### Requirements

* Python 3.11+
* [`uv`](https://docs.astral.sh/uv/)
* A Weights & Biases account if you want to reproduce experiment tracking

Clone the repository and install dependencies:

```bash
git clone https://github.com/aaronnn05/F1-Apex-Pace.git
cd F1-Apex-Pace

uv sync
```

### Recreate the Model

The trained model is not committed to the repository. You can reproduce it by running the project pipeline.

First, build the processed dataset:

```bash
uv run python scripts/build_pipeline.py
```

This downloads the required Formula 1 data through FastF1, stores the raw data in DuckDB, performs feature engineering, and writes the processed features to:

```text
data/processed/features_v1.parquet
```

Then run the model training and tuning:

```bash
uv run python scripts/run_tuning.py
```

This trains the XGBoost model, runs Optuna hyperparameter tuning, evaluates the tuned model, and saves the resulting model to:

```text
models/xgboost_tuned.json
```

Once this file exists, the FastAPI application can load the model.
I.e. by 
```bash
uv run pytest
uv run uvicorn src.apex_pace.api.main:app --reload
```

> **Note:** FastF1 downloads race data from external sources, so the pipeline may take some time to complete on the first run.

## Data & Feature Engineering

Race data is collected with **FastF1**, stored through **DuckDB**, and transformed using **Polars**.

Features are generated sequentially within each driver/race to avoid using future lap information.

| Feature             | Description                                  |
| ------------------- | -------------------------------------------- |
| `prev_lap_time`     | Previous completed lap time                  |
| `rolling_3lap_mean` | Rolling mean of recent lap times             |
| `compound_code`     | Encoded tyre compound                        |
| `is_fresh_tyre`     | Whether the current tyre is fresh            |
| `TyreLife`          | Number of laps completed on the current tyre |

Pit laps and unsuitable track-condition observations are removed before modelling.

## Temporal Validation

A random train/test split would allow future race information to influence training.

Instead, the project uses chronological splits:

* **Training:** Bahrain 2023 laps 4–46
* **Validation:** Bahrain 2023 laps 47–57
* **Test:** Bahrain 2024

The final model is retrained on the full 2023 training + validation data before evaluation on 2024.

## Model Experiments

Mean Absolute Error (MAE) and Root Mean Squared Error (RMSE) are used to evaluate predictions.

| Model                        |    MAE (s) |   RMSE (s) |
| ---------------------------- | ---------: | ---------: |
| Mean baseline                |     1.7405 |     2.0667 |
| Previous-lap baseline        | **0.3790** |     0.6714 |
| XGBoost + MSE                |     0.5291 |     0.7426 |
| XGBoost + Pseudo-Huber       |     0.4765 |     0.6717 |
| Tuned XGBoost + Pseudo-Huber |     0.4569 | **0.6465** |

### Key Findings

The previous-lap baseline was surprisingly strong, outperforming the XGBoost models on MAE.

This showed that additional model complexity does not automatically improve performance when a strong domain-specific baseline already captures most of the signal.

Hyperparameter tuning improved the XGBoost model, particularly in reducing larger prediction errors. Slice analysis showed improved RMSE and tail errors for HARD-tyre laps compared with the untuned model.

The result also highlighted an opportunity for future work: **better feature engineering may provide more value than further hyperparameter tuning.**

## Model Tuning

XGBoost hyperparameters were optimized with **Optuna's TPE sampler** using the chronological validation set.

The optimization objective was validation MAE.

The final model uses the `reg:pseudohubererror` objective, which was selected to provide more robustness to larger lap-time errors than standard squared-error training.

Experiment tracking was performed with **Weights & Biases**.

## API

A FastAPI service exposes the trained model through a prediction endpoint.

### Start the API

After recreating the model:

```bash
uv run uvicorn src.apex_pace.api.main:app --reload
```

Interactive API documentation is available at:

```text
http://127.0.0.1:8000/docs
```

### Example request

```json
{
  "prev_lap_time": 98.5,
  "rolling_3lap_mean": 98.6,
  "compound_code": 3,
  "is_fresh_tyre": 0,
  "TyreLife": 15
}
```

### Example response

```json
{
  "predicted_lap_time_seconds": 98.742,
  "model_version": "xgboost_tuned_v1"
}
```

The API uses Pydantic validation to reject invalid feature values before they reach the model.

## Testing

The API is tested with `pytest` and FastAPI's `TestClient`.

Current tests cover:

* Health endpoint
* Successful prediction
* Invalid tyre life
* Invalid tyre compound

Run the tests with:

```bash
uv run pytest
```

## Tech Stack

**Data**

* FastF1
* DuckDB
* Polars
* Parquet

**Machine Learning**

* XGBoost
* Scikit-learn
* Optuna

**MLOps**

* Weights & Biases

**API**

* FastAPI
* Pydantic
* Uvicorn

**Testing & Tooling**

* pytest
* uv

## Project Structure

```text
F1-Apex-Pace/
├── src/
│   └── apex_pace/
│       ├── api/
│       ├── data/
│       ├── features/
│       └── models/
├── scripts/
│   ├── build_pipeline.py
│   ├── run_baselines.py
│   └── run_tuning.py
├── tests/
│   └── test_api.py
├── data/
├── models/
├── pyproject.toml
├── uv.lock
└── README.md
```

## Future Improvements

Potential next steps include:

* Adding circuit and driver features
* Incorporating weather and track-condition variables
* Improving tyre-degradation features
* Evaluating across multiple circuits and seasons
* Containerizing the API
