"""Polars feature transformation pipeline with point-in-time correctness."""

from pathlib import Path

import fastf1
import polars as pl


def get_laps_data():

    # 1. Setup local cache directory
    CACHE_DIR = Path(__file__).resolve().parents[3] / "data" / "raw" / "fastf1_cache"
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    fastf1.Cache.enable_cache(CACHE_DIR)

    print("FastF1 cache enabled successfully.")

    # 2. Load 2024 Bahrain Grand Prix Race Session
    # Year: 2024, Grand Prix: 'Bahrain', Session: 'R' (Race)
    session = fastf1.get_session(2024, "Bahrain", "R")
    session.load(telemetry=False, weather=True)

    print("Race session data loaded")

    # 3. Extract laps dataframe
    laps = session.laps

    # 4. Convert Pandas df into Polars df
    # Polars allow high-performance transformations
    laps_pl = pl.from_pandas(laps)

    # Convert LapTime from timedelta object to seconds
    laps_pl = laps_pl.with_columns((pl.col("LapTime").dt.total_milliseconds() / 1000).alias("LapTimeSeconds"))

    return laps_pl


def build_feature_set(laps_df: pl.DataFrame) -> pl.DataFrame:
    """
    Transform raw race laps into point-in-time features for machine learning.
    """
    # TODO 1: Filter out noisy/invalid laps based on our EDA findings.
    # Requirements:
    # - Keep only green flag laps (`TrackStatus == '1'`).
    # - Drop laps where `LapTimeSeconds` is null.
    # - Drop in-laps and out-laps (keep rows where `PitInTime` IS NULL AND `PitOutTime` IS NULL).
    #
    # --- WRITE YOUR TODO 1 CODE HERE ---
    clean_laps = ...

    # TODO 2: Ensure strictly sequential ordering before applying window operations.
    # Requirements: Sort by `["Year", "EventName", "Driver", "LapNumber"]`.
    #
    # --- WRITE YOUR TODO 2 CODE HERE ---
    sorted_laps = ...

    # TODO 3: Compute Point-in-Time Features using Polars Windowing.
    # Requirements:
    # 1. `prev_lap_time`: Shift `LapTimeSeconds` by 1 grouped over `["Year", "EventName", "Driver"]`.
    # 2. `rolling_3lap_mean`: Take rolling mean of window_size=3 (min_periods=1) on shifted lap time grouped by `["Year", "EventName", "Driver"]`.
    # 3. `compound_code`: Replace Compound strings ("SOFT", "MEDIUM", "HARD") with integer IDs (1, 2, 3), default 0, cast to pl.Int32.
    # 4. `is_fresh_tyre`: Cast `FreshTyre` boolean to pl.Int32.
    #
    # --- WRITE YOUR TODO 3 CODE HERE ---
    features_df = sorted_laps.with_columns(
        [
            # Fill in expressions here...
        ]
    )

    # TODO 4: Drop initial rows where `prev_lap_time` is null (since Lap 1 has no previous lap time).
    #
    # --- WRITE YOUR TODO 4 CODE HERE ---
    final_features = ...

    return final_features
