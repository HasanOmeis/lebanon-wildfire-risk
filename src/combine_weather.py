from pathlib import Path  # Path builds file paths that work on any OS
import pandas as pd  # pandas loads, merges and saves the tables
from weather import RAW_DIR, MAP_PATH  # reuse the batch folder and cell-to-point lookup from weather.py

OUT_PATH = Path("data/processed/weather_daily.parquet")  # final weather table, one row per cell per day


def load_batches():  # stack every downloaded batch into one table
    files = sorted(RAW_DIR.glob("weather_*.parquet"))  # all batch files, in name order
    print("Batch files found:", len(files))  # 90 when the download is complete
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)  # read each file and stack them
    df = df.drop_duplicates(subset=["wp_id", "date"])  # safety: one row per point per day
    return df  # weather per weather point


def to_cells(points_weather):  # copy each point's weather to every cell that uses it
    lookup = pd.read_csv(MAP_PATH)  # cell_id → wp_id table from step 3
    cells = lookup.merge(points_weather, on="wp_id", how="inner")  # one row per cell per day
    cells = cells.drop(columns=["wp_id"])  # the point ID is no longer needed
    cells = cells.sort_values(["cell_id", "date"]).reset_index(drop=True)  # tidy order: by cell, then by day
    return cells  # weather per 5 km cell


if __name__ == "__main__":  # only run this block when the file is executed directly
    pw = load_batches()  # weather per point
    print("Point rows:", len(pw))  # points × days downloaded so far
    weather = to_cells(pw)  # weather per cell
    print("Cell rows:", len(weather))  # 2,631,296 when complete
    print("Cells:", weather["cell_id"].nunique())  # 488 once every point group is in
    print("Dates:", weather["date"].min().date(), "to", weather["date"].max().date())  # range covered so far
    print("Missing values per column:")  # header for the next print
    print(weather.isna().sum())  # should all be 0, or close to it
    print(weather.dtypes)  # date should be datetime64, the weather columns float64
    weather.to_parquet(OUT_PATH, index=False)  # save the combined table
    print(f"Saved {OUT_PATH}")  # confirm the save