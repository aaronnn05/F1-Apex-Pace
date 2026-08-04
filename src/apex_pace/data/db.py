"""DuckDB database connection and table initialisation wrappers"""

from pathlib import Path

import duckdb

DEFAULT_DB_PATH = Path(__file__).resolve().parents[3] / "data" / "f1_vault.duckdb"

class F1Database:
    """Manager for DuckDB storage operations."""

    def __init__(self, db_path: Path = DEFAULT_DB_PATH):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

    def get_connection(self) -> duckdb.DuckDBPyConnection:
        """
        Return an active DuckDB connection object.
        """
        conn = duckdb.connect(str(self.db_path))
        return conn

    def init_tables(self) -> None:
        """
        Create the `raw_laps` table schema if it does not already exist.
        """
        with self.get_connection() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS raw_laps (
                    Year INTEGER,
                    EventName VARCHAR,
                    Driver VARCHAR,
                    LapNumber INTEGER,
                    LapTimeSeconds DOUBLE,
                    Compound VARCHAR,
                    TyreLife INTEGER,
                    FreshTyre BOOLEAN,
                    Stint INTEGER,
                    TrackStatus VARCHAR,
                    PitInTime VARCHAR,
                    PitOutTime VARCHAR,
                    Team VARCHAR,
                    TrackTemp DOUBLE,
                    AirTemp DOUBLE,
                    PRIMARY KEY (Year, EventName, Driver, LapNumber)
                )
            """)
            print(f"DuckDB initialised at: {self.db_path.resolve()}")

if __name__ == "__main__":
    db = F1Database()
    db.init_tables()
    with db.get_connection() as conn:
        tables = conn.execute("SHOW TABLES").fetchall()
        print("Current tables in DuckDB", tables)