"""Master script to ingest raw data into DuckDB and output processed Parquet features."""

from pathlib import Path

import fastf1
import polars as pl

from apex_pace.data.db import F1Database
from apex_pace.features.pipeline import build_feature_set

# 1. Setup local cache directory
BASE_DIR = Path(__file__).resolve().parents[3]
CACHE_DIR = BASE_DIR / "data" / "raw" / "fastf1_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
fastf1.Cache.enable_cache(CACHE_DIR)

SEASONS_TO_INGEST = [
    (2023, "Bahrain"),
    (2024, "Bahrain")
]

def ingest_and_process():
    db = F1Database()
    db.init_tables()

    # Ingest raw sessions into DuckDB
    for year, event in SEASONS_TO_INGEST:
        session = fastf1.get_session(year, event, "R")
        session.load(telemetry=False, weather=True)

        raw_df = pl.from_pandas(session.laps)

        transformed_df = raw_df.with_columns([
            (pl.col("LapTime").dt.total_milliseconds() / 1000.0).alias("LapTimeSeconds"),
            pl.lit(year).alias("Year"),
            pl.lit(event).alias("EventName"),
            pl.col("PitInTime").dt.total_milliseconds().cast(pl.String).alias("PitInTime"),
            pl.col("PitOutTime").dt.total_milliseconds().cast(pl.String).alias("PitOutTime"),
            pl.lit(None).cast(pl.Float64).alias("TrackTemp"),
            pl.lit(None).cast(pl.Float64).alias("AirTemp")
        ]).select([
            "Year", "EventName", "Driver", "LapNumber", "LapTimeSeconds",
            "Compound", "TyreLife", "FreshTyre", "Stint", "TrackStatus",
            "PitInTime", "PitOutTime", "Team", "TrackTemp", "AirTemp"
        ])

        with db.get_connection() as conn:
            conn.execute("DELETE FROM raw_laps WHERE Year = ? AND EventName = ?", [year, event])
            conn.register("temp_df", transformed_df)
            conn.execute("INSERT INTO raw_laps SELECT * FROM temp_df")

    # Query DuckDB for all raw laps
    with db.get_connection() as conn:
        all_laps = conn.execute("SELECT * FROM raw_laps").pl()

    # pass through "build_feature_set()" to clean/feature engineer it
    processed_all_laps = build_feature_set(all_laps)

    # and save to "data/processed/features_v1.parquet".    
    output_dir = BASE_DIR / "data" / "processed"
    output_dir.mkdir(parents=True, exist_ok=True)
    processed_all_laps.write_parquet(output_dir / "features_v1.parquet")

    print("Success! Feature matrix written to", output_dir / "features_v1.parquet")
    print(processed_all_laps.schema)


if __name__ == "__main__":
    ingest_and_process()