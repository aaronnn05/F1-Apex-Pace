"""Polars feature transformation pipeline with point-in-time correctness."""

import polars as pl


def build_feature_set(laps_df: pl.DataFrame) -> pl.DataFrame:
    """
    Transform raw race laps into point-in-time features for machine learning.
    """
    clean_laps = laps_df.filter(
        (pl.col("TrackStatus") == "1") &
        (pl.col("LapTimeSeconds").is_not_null()) &
        (pl.col("PitInTime").is_null()) &
        (pl.col("PitOutTime").is_null())
    )

    # Sort sequentially before windowing
    sorted_laps = clean_laps.sort(["Year", "EventName", "Driver", "LapNumber"])

    # Compute Point-in-Time Features using Polars Windowin.
    features_df = sorted_laps.with_columns(
        [
            pl.col("LapTimeSeconds").shift(1).over(["Year", "EventName", "Driver"]).alias("prev_lap_time"),
            pl.col("LapTimeSeconds").shift(1).rolling_mean(window_size=3, min_periods=1).over(["Year", "EventName", "Driver"]).alias("rolling_3lap_mean"),
            pl.col("Compound").replace({
                "SOFT": 1,
                "MEDIUM": 2,
                "HARD": 3,
                "INTERMEDIATE": 4,
                "WET": 5            
            }).cast(pl.Int32).alias("compound_code"),
            pl.col("FreshTyre").cast(pl.Int32).alias("is_fresh_tyre")
        ]
    )

    # Drop initial rows where `prev_lap_time` is null (since Lap 1 has no previous lap time).
    final_features = features_df.filter(pl.col("prev_lap_time").is_not_null())

    return final_features
