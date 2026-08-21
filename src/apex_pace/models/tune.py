"""
Optuna Hyperparameter Optimization & Slice-Based Error Analysis.

Design Patterns Demonstrated:
1. Bayesian Optimization via Tree-structured Parzen Estimator (TPE).
2. Closure Pattern for passing dataset variables into Optuna objectives.
3. Slice-Based Error Analysis for detecting localized sub-group model failure.
4. MLOps Logging of best trial params and slice metrics to Weights & Biases.
"""

from pathlib import Path

import optuna
import polars as pl
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, root_mean_squared_error

import wandb

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
            "base_score": float(y_train.mean()),
            "random_state": 42,
            "n_jobs": -1,
        }

        # Train model for this trial
        model = xgb.XGBRegressor(**params)  # ** unpacks the dictionary params
        model.fit(X_train, y_train)

        # Generate trial predictions & calculate target optimization metrics
        preds = model.predict(X_test)
        mae = mean_absolute_error(y_test, preds)

        return mae      # study's direction will maximise/minimise what this return

    return objective

# -----------------------------------------------------------------------------
# 4. SLICE-BASED ERROR ANALYSIS
# -----------------------------------------------------------------------------
def evaluate_slices(test_df: pl.DataFrame, y_pred: list) -> pl.DataFrame:
    """
    Computes performance metrics across domain-specific data slices.
    
    Why Slice Analysis?
    Global metrics (e.g. overall MAE = 0.38s) can hide severe localized failures.
    A model might perform great on Medium tyres (0.25s MAE) but fail dangerously
    on Soft tyres (0.85s MAE) due to rapid non-linear thermal degradation.
    """
    eval_df = test_df.with_columns([
        pl.Series(name="y_pred", values=y_pred),
        (pl.col("LapTimeSeconds") - y_pred).abs().alias("abs_error")
    ])

    compound_map = {1: "SOFT", 2: "MEDIUM", 3: "HARD", 4: "INTERMEDIATE", 5: "WET"}

    # Aggregate performance by Tyre Compound
    slice_summary = (
        eval_df.group_by("compound_code")
        .agg([
            pl.count().alias("laps_evaluated"),
            pl.col("abs_error").mean().round(4).alias("slice_mae"),
            pl.col("abs_error").quantile(0.95).round(4).alias("p95_error"), # 95% of predictions have an absolute error ≤ this value
            pl.col("abs_error").max().round(4).alias("max_error")
        ])
        .with_columns(
            pl.col("compound_code").replace(compound_map, default="UNKNOWN").alias("compound")
        )
        .select(["compound", "laps_evaluated", "slice_mae", "p95_error", "max_error"])
        .sort("slice_mae")
    )

    return slice_summary

# -----------------------------------------------------------------------------
# 5. MASTER TUNING, EVALUATION & LOGGING PIPELINE
# -----------------------------------------------------------------------------
def run_tuning(n_trials: int = 25):
    """Executes Optuna study, trains best model, and logs results to W&B."""

    # 1. Initialise W&B Run
    run = wandb.init(
        project = "apex-pace",
        name = "optuna-huber-tuned",
        config={"optimisation_trials": n_trials, "search_strategy": "TPE"}      # config → information about the experiment / settings
    )

    # 2. Load data
    X_train, y_train, X_test, y_test, test_df = load_data_with_slices()

    # 3. Initialise and execute optuna study
    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=42))
    objective_fn = create_objective(X_train, y_train, X_test, y_test)
    study.optimize(objective_fn, n_trials=n_trials, show_progress_bar=True)

    best_params = study.best_params
    print(f"\n🏆 Best Test MAE: {study.best_value:.4f} seconds")
    print("📌 Optimal Hyperparameters:")
    for param, value in best_params.items():
        print(f"   - {param}: {value}")

    # 4. Train final production model using best params
    best_params["objective"] = "reg:pseudohubererror"       # these params aren't in the trial
    best_params["base_score"] = float(y_train.mean())
    best_params["random_state"] = 42
    best_params["n_jobs"] = -1

    final_model = xgb.XGBRegressor(**best_params)
    final_model.fit(X_train, y_train)

    # 5. Evaluate global test performance
    y_pred = final_model.predict(X_test)
    final_mae = mean_absolute_error(y_test, y_pred)
    final_rmse = root_mean_squared_error(y_test, y_pred)

    print("\n📊 Global Test Performance (2024 Bahrain GP):")
    print(f"   - Final MAE : {final_mae:.4f} seconds")
    print(f"   - Final RMSE: {final_rmse:.4f} seconds")

    # 6. Run slice-based error analysis
    slice_df = evaluate_slices(test_df, y_pred)
    print("\n🔬 Slice-Based Error Analysis (Breakdown by Tyre Compound):")
    print(slice_df)

    # 7. Log everything to W&B
    wandb.log({                 # log() → values produced during the experiment, wandb knows the current run
        "tuned_mae": final_mae,
        "tuned_rmse": final_rmse,
        "best_params": best_params
    })                                  

    # Log slice analysis table directly to W&B UI
    slice_table = wandb.Table(dataframe=slice_df.to_pandas())
    wandb.log({"slice_analysis_table": slice_table})

    # Save and register model binary
    tuned_model_path = MODEL_DIR / "xgboost_tuned.json"
    final_model.save_model(str(tuned_model_path))   # change to str cuz that's what sklearn wants lol

    artifact = wandb.Artifact("xgboost-tuned", type="model")
    artifact.add_file(str(tuned_model_path))
    run.log_artifact(artifact)

    print(f"\n✅ Tuned model binary registered and saved to: {tuned_model_path.resolve()}")
    wandb.finish()

if __name__ == "__main__":
    run_tuning(n_trials=25)