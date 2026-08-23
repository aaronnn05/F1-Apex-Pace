"""Training pipeline for XGBoost tyre degradation model with W&B logging."""

from pathlib import Path

import polars as pl
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, root_mean_squared_error

import wandb

BASE_DIR = Path(__file__).resolve().parents[3]
DATA_PATH = BASE_DIR / "data" / "processed" / "features_v1.parquet"
MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

def load_and_split_data(event_name: str, data_path: Path = DATA_PATH):
    """
    Loads feature matrix from Parquet and executes a temporal train/validation split.
    
    Why Temporal Split?
    In time-series/event data, random K-Fold cross-validation leaks future information 
    into the past. We strictly train on past seasons (2023) and validate on future seasons (2024).
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

    train_df = df.filter(
            pl.col("Year") == 2023, 
            pl.col("EventName") == event_name
            ).sort(["LapNumber", "Driver"])
    
    test_df = df.filter(
            pl.col("Year") == 2024,
            pl.col("EventName") == event_name    
            ).sort(["LapNumber", "Driver"])

    # Convert Polars selections to NumPy/Pandas structures expected by XGBoost & Scikit-Learn
    X_train = train_df.select(feature_cols).to_pandas()
    # .values.ravel() flattens 2D DataFrame column into 1D array required for regression targets
    y_train = train_df.select(target_col).to_pandas().values.ravel()

    X_test = test_df.select(feature_cols).to_pandas()
    y_test = test_df.select(target_col).to_pandas().values.ravel()

    return X_train, y_train, X_test, y_test 


def train_and_log():
    """Executes model training, evaluates metrics, and logs metadata to W&B."""

    X_train, y_train, X_test, y_test = load_and_split_data("Bahrain", DATA_PATH)

    # Initialize Weights & Biases Run for tracking
    run = wandb.init(
            project="apex-pace",
            name="xgboost-huber-baseline",
            config={
                "model_type": "XGBoost",
                "objective": "reg:pseudohubererror",
                "base_score": float(y_train.mean()),
                "n_estimators": 100,
                "learning_rate": 0.05,
                "max_depth":5,
                "random_state": 42
            }
        )
    config = wandb.config       # wandb.config saved whatever in the run's config

    model = xgb.XGBRegressor(
        n_estimators=config.n_estimators,
        learning_rate=config.learning_rate,
        max_depth=config.max_depth,
        objective=config.objective,
        random_state=config.random_state,
        base_score=config.base_score
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    # Compute evaluation metrics
    mae = mean_absolute_error(y_test, y_pred)
    rmse = root_mean_squared_error(y_test, y_pred)

    print("\n📊 Validation Results (2024 Bahrain GP):")
    print(f"   - Mean Absolute Error (MAE) : {mae:.4f} seconds")
    print(f"   - Root Mean Squared Error (RMSE): {rmse:.4f} seconds")

    # Log metrics to W&B cloud dashboard
    wandb.log({"test_mae": mae, "test_rmse": rmse})

    # Save model locally first (merely so there's a file to upload to W&B)
    local_model_path = MODEL_DIR / "xgboost_baseline.json"
    model.save_model(str(local_model_path))
    print(f"\n✅ Local model artifact written to: {local_model_path}")

    # Register saved model binary as a versioned W&B artifact
    artifact = wandb.Artifact("xgboost-baseline", type="model")
    artifact.add_file(str(local_model_path))
    run.log_artifact(artifact)

    # Finalise W&B run cleanly
    wandb.finish()

    