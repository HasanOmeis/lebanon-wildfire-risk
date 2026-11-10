"""Settings for the daily pipeline: paths, area, fetch windows and the FIRMS key."""  # file description

import datetime as dt  # dates for "today" and the fetch window
import os  # reads environment variables (the FIRMS key)
from pathlib import Path  # file paths that work on any OS

from dotenv import load_dotenv  # loads .env into environment variables

ROOT = Path(__file__).resolve().parents[2]  # project root (src/pipeline/config.py -> two folders up)
load_dotenv(ROOT / ".env")  # read .env from the project root, wherever the script is started from

# ---- folders ----
RAW = ROOT / "data" / "raw"  # raw downloads (Day 2)
PROCESSED = ROOT / "data" / "processed"  # cleaned tables (Days 3-7)
SERVING = ROOT / "data" / "serving"  # small files the API reads (Day 13)
PREDICTIONS = ROOT / "data" / "predictions"  # one dated prediction file per day (Step 1)
MODELS = ROOT / "models"  # tuned model and metadata (Days 11-13)
RUNS = ROOT / "pipeline_runs"  # one log file per pipeline run (Step 1)

# ---- input files the pipeline updates or reads ----
FIRES_RAW = RAW / "firms_lebanon.parquet"  # cleaned fire detections (Day 2)
FIRES_CELLS = PROCESSED / "fires_with_cells.parquet"  # fires with their cell_id (Day 3)
GRID = PROCESSED / "lebanon_grid.gpkg"  # the 488 grid cells (Day 3)
WEATHER = PROCESSED / "weather_daily.parquet"  # weather per cell per day (Day 4)
WEATHER_POINTS = PROCESSED / "weather_points.csv"  # the 0.1° points we ask Open-Meteo about (Day 4)
CELL_TO_POINT = PROCESSED / "cell_to_weather_point.csv"  # which weather point each cell uses (Day 4)
NDVI = PROCESSED / "ndvi.parquet"  # NDVI per cell per composite (Days 5-6)
TERRAIN = PROCESSED / "terrain.parquet"  # static terrain per cell (Day 5)
FEATURES = PROCESSED / "features.parquet"  # full feature table (Day 7)
MODEL_META = MODELS / "model_meta.json"  # the model's feature list (Day 13)

# ---- area and sources ----
BBOX = "35.10,33.05,36.63,34.70"  # Lebanon box: west,south,east,north (same as Day 2)
FIRMS_SOURCE = "VIIRS_SNPP_NRT"  # near-real-time VIIRS: the only source with yesterday's fires
FIRMS_URL = "https://firms.modaps.eosdis.nasa.gov/api/area/csv"  # FIRMS "area" endpoint (same as Day 2)

# ---- fetch windows ----
FIRMS_DAYS = 5  # re-fetch the last 5 days each run (the API max per request), so late detections are caught
WEATHER_PAST_DAYS = 7  # re-fetch the last 7 days of weather, so a missed run is filled automatically
EXPECTED_CELLS = 488  # every day must have exactly this many rows (Day 3 grid)


def today_utc() -> dt.date:  # FIRMS and Open-Meteo work in UTC, so the pipeline does too
    return dt.datetime.now(dt.timezone.utc).date()  # today's date in UTC


def firms_key() -> str:  # read the key only when a step needs it (tests can run without it)
    key = os.getenv("FIRMS_MAP_KEY")  # same variable name as Day 2
    if not key:  # missing or empty
        raise RuntimeError("FIRMS_MAP_KEY not found. Check the .env file in the project root.")  # clear message
    return key  # the key itself (never print it)


if __name__ == "__main__":  # quick self-check: python -m src.pipeline.config
    for name, path in [  # every file the pipeline depends on
        ("fires raw", FIRES_RAW), ("fires cells", FIRES_CELLS), ("grid", GRID),  # fire and grid files
        ("weather", WEATHER), ("weather points", WEATHER_POINTS), ("cell->point", CELL_TO_POINT),  # weather files
        ("ndvi", NDVI), ("terrain", TERRAIN), ("features", FEATURES), ("model meta", MODEL_META),  # the rest
    ]:  # end of the list
        print(f"{'OK     ' if path.exists() else 'MISSING'} {name:15} {path.relative_to(ROOT)}")  # status, name, short path
    print("FIRMS key found:", bool(os.getenv("FIRMS_MAP_KEY")))  # True/False only, never the key
    print("Today (UTC):", today_utc())  # the date the pipeline will treat as "today"