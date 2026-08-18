"""
Optuna Hyperparameter Optimization & Slice-Based Error Analysis.

Design Patterns Demonstrated:
1. Bayesian Optimization via Tree-structured Parzen Estimator (TPE).
2. Closure Pattern for passing dataset variables into Optuna objectives.
3. Slice-Based Error Analysis for detecting localized sub-group model failure.
4. MLOps Logging of best trial params and slice metrics to Weights & Biases.
"""

from pathLib import Path
import optuna
import polars as pl
import xgboost as xgb
import wandb
from sklearn.metrics import mean_absolute_error, root_mean_squared_error

# Supress verbose Optuna logs, only show warnings
optuna.logging.set_verbosity(optuna.logging.WARNING)

# -----------------------------------------------------------------------------
# 1. DIRECTORY RESOLUTION & SETUP
# -----------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parents[3]
DATA_PATH = BASE_DIR / "data" / "processed" / "features_v1.parquet"
MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)