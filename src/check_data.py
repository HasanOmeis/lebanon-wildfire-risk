from pathlib import Path  # clean file paths that work on Windows

import geopandas as gpd  # read the grid file (.gpkg)
import pandas as pd  # read and summarize the tables

FIRES_PATH = Path("data/processed/fires_with_cells.parquet")  # Day 3: fires with their cell_id
GRID_PATH = Path("data/processed/lebanon_grid.gpkg")  # Day 3: the 488 cells
TERRAIN_PATH = Path("data/processed/terrain.parquet")  # Day 5: per-cell terrain stats
NDVI_DIR = Path("data/raw/ndvi")  # Day 5: one NDVI file per date
WEATHER_DIR = Path("data/raw/weather")  # Day 4: one weather file per year-group batch
NDVI_TOTAL = 663  # the real number of NDVI dates (539 was only the resumed part)
WEATHER_TOTAL = 90  # the number of batches the weather script planned


def date_range(df):  # find a date column and return its first and last day
    for col in df.columns:  # look at each column name
        if "date" in col.lower() or "time" in col.lower():  # column names like acq_date, end_date, time
            dates = pd.to_datetime(df[col], errors="coerce")  # convert to dates; anything that isn't a date becomes NaT
            if dates.notna().any():  # only use it if it actually contains dates
                return f"{col}: {dates.min().date()} -> {dates.max().date()}"  # e.g. acq_date: 2012-01-20 -> 2026-10-30
    return "no date column"  # static tables like terrain have no dates


def summarize(name, df):  # print one block of facts about a table
    print(f"\n=== {name} ===")  # header so each dataset is easy to spot
    print("Shape (rows, cols):", df.shape)  # size of the table
    print("Columns:", list(df.columns))  # column names, so we know what to use in the notebook
    print("Dates:", date_range(df))  # time coverage
    missing = df.isna().sum()  # count missing values per column
    missing = missing[missing > 0]  # keep only columns that have some missing values
    print("Missing values:", dict(missing) if len(missing) else "none")  # show them, or say none


def read_folder(folder):  # read every parquet file in a folder into one table
    files = sorted(folder.glob("*.parquet"))  # list the files, in name order
    if not files:  # nothing downloaded yet
        return files, pd.DataFrame()  # return an empty table
    return files, pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)  # stack all files into one table


if __name__ == "__main__":  # runs only when this file is run directly
    summarize("Fires", pd.read_parquet(FIRES_PATH))  # fire detections
    summarize("Grid", gpd.read_file(GRID_PATH).drop(columns="geometry"))  # cells without the shapes (just the attributes)
    summarize("Terrain", pd.read_parquet(TERRAIN_PATH))  # terrain stats

    ndvi_files, ndvi = read_folder(NDVI_DIR)  # all NDVI dates so far
    print(f"\nNDVI files: {len(ndvi_files)} of {NDVI_TOTAL}")  # how far the NDVI download got
    summarize("NDVI", ndvi)  # NDVI table

    weather_files, weather = read_folder(WEATHER_DIR)  # all weather batches so far
    print(f"\nWeather files: {len(weather_files)} of {WEATHER_TOTAL}")  # how far the weather download got
    summarize("Weather", weather)  # weather table