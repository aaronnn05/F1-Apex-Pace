from pathlib import Path

import fastf1
import polars as pl

# 1. Setup local cache directory
CACHE_DIR = Path("data/raw/fastf1_cache")
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
