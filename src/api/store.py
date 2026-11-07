"""Loads the serving files once: recent features grouped by day, and the grid as GeoJSON."""  # file description

import datetime as dt  # date type for lookups
import json  # reads grid.geojson
from pathlib import Path  # file paths

import pandas as pd  # reads the features file

ROOT = Path(__file__).resolve().parents[2]  # project root (src/api/store.py -> two folders up)
SERVING_DIR = ROOT / "data" / "serving"  # built by src/export_serving.py


class FeatureStore:  # holds the serving data in memory
    def __init__(self, serving_dir: Path = SERVING_DIR):  # runs once, at API startup
        df = pd.read_parquet(serving_dir / "recent_features.parquet")  # last 400 days x 488 cells
        df["date"] = pd.to_datetime(df["date"]).dt.date  # plain dates, to match the dates FastAPI gives us
        self.by_day = {day: rows.reset_index(drop=True) for day, rows in df.groupby("date")}  # {date: that day's 488 rows}
        self.first = min(self.by_day)  # oldest day available
        self.last = max(self.by_day)  # newest day available
        self.grid = json.loads((serving_dir / "grid.geojson").read_text(encoding="utf-8"))  # the 488 polygons as a GeoJSON dict

    def rows_for(self, day: dt.date) -> pd.DataFrame | None:  # all cells on one day
        return self.by_day.get(day)  # None if the day isn't in the window

    def cell_row(self, day: dt.date, cell_id: int) -> pd.DataFrame | None:  # one cell on one day (used by /explain in Step 10)
        rows = self.rows_for(day)  # that day's cells
        if rows is None:  # day not available
            return None  # nothing to return
        match = rows[rows["cell_id"] == cell_id]  # keep only the asked cell
        return None if match.empty else match.reset_index(drop=True)  # None if the cell doesn't exist