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

# -----------------------------------------------------------------------------
# 2. DATA EXTRACTION WITH SLICE METADATA RETENTION
# -----------------------------------------------------------------------------
def load_data_with_slices(data_path: Path = DATA_PATH, target_event: str = "Bahrain"):
    """
    Loads dataset, performs temporal split, and returns evaluation DataFrame
    with raw categorical columns intact for slice-based error analysis.
    """
    df = pl.read_parquet(data_path)

    feature_cols = [
            "prev_lap_time",
            "rolling_3lap_mean",
            "compound_code",
            "is_fresh_tyre",
            "TyreLife"
        ]

    target_col = "LapTimeSeconds"

    # Temporal Split: 2023 data for train, 2024 for test
    train_df = df.filter(pl.col("Year") == 2023, pl.col("EventName") == target_event)
    test_df = df.filter(pl.col("Year") == 2024, pl.col("EventName") == target_event)

    # Feature and Target extractions
    X_train = train_df.select(feature_cols).to_pandas()
    y_train = train_df.select(target_col).to_pandas().values.ravel()

    X_test = test_df.select(feature_cols).to_pandas()
    y_test = test_df.select(target_col).to_pandas().values.ravel()

    return X_train, y_train, X_test, y_test, test_df

# -----------------------------------------------------------------------------
# 3. OPTUNA OBJECTIVE CLOSURE
# -----------------------------------------------------------------------------
def create_objective(X_train, y_train, X_test, y_test):
    """
    Closure pattern: Returns an objective function parameterized with train test df.
    
    Why use a closure?
    Optuna's `study.optimize` requires a callable with signature `objective(trial)`.
    Wrapping it inside this factory function gives the inner `objective` access
    to `X_train`, `y_train`, `X_test`, and `y_test` without global variables,
    i.e. a reusable function + remembered variables within create_objective's scope.
    """
    def objective(trial: optuna.Trial) -> float:
        params = {
            # Tree architecture controls
            "n_estimators": trial.suggest_int("n_estimators", 50, 300, step=25),
            "max_depth": trial.suggest_int("max_depth", 3, 8),
            # Log-uniform distribution for learning rate (searches across orders of magnitude)
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.20, log=True),
            # Stochastic subsampling controls (regularization against overfitting)
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            # L1 (Lasso) and L2 (Ridge) leaf weight regularization
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
            # Fixed settings
            "objective": "reg:pseudohubererror",
            "random_state": 42,
            "n_jobs": -1,
        }

        # Train model for this trial
        model = xgb.XGBRegressor(**params)  # ** unpacks the dictionary params
        model.fit(X_train, y_train)

        # Generate trial predictions & calculate target optimization metrics
        preds = model.predict(X_test)
        mae = mean_absolute_error(y_test, preds)

        return mae

    return objective
