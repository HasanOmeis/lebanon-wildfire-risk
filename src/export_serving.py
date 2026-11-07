"""Builds the small serving files the API reads: grid.geojson and recent_features.parquet."""  # file description

import json  # reads model_meta.json
from pathlib import Path  # file paths

import geopandas as gpd  # reads and reprojects the grid
import pandas as pd  # reads and filters the feature table

ROOT = Path(__file__).resolve().parents[1]  # project root (src/export_serving.py -> one folder up)
PROCESSED = ROOT / "data" / "processed"  # training-side data (Days 3-7)
SERVING = ROOT / "data" / "serving"  # serving-side data: only what the API needs
DAYS_KEPT = 400  # a bit more than a year of days, so the map can show any recent date


def export_grid() -> int:  # write grid.geojson in lat/lon; returns the number of cells
    grid = gpd.read_file(PROCESSED / "lebanon_grid.gpkg")  # the 488 cells from Day 3 (UTM meters)
    centers = grid.geometry.centroid.to_crs(4326)  # cell centers computed in meters (accurate), then turned into lat/lon
    grid = grid.to_crs(4326)  # polygons to lat/lon (EPSG:4326), what web maps expect
    grid["lat"] = centers.y.round(5)  # center latitude, 5 decimals (about 1 m)
    grid["lon"] = centers.x.round(5)  # center longitude
    grid["land_frac"] = grid["land_frac"].round(3)  # share of the cell on land (coast cells are partly sea)
    grid["geometry"] = grid.geometry.set_precision(1e-5)  # round corner coordinates to about 1 m: a much smaller file
    out = grid[["cell_id", "land_frac", "lat", "lon", "geometry"]]  # only the columns the map needs
    out.to_file(SERVING / "grid.geojson", driver="GeoJSON")  # save as GeoJSON
    return len(out)  # cell count for the printout


def export_features() -> pd.DataFrame:  # write the last DAYS_KEPT days of features, without the label
    meta = json.loads((ROOT / "models" / "model_meta.json").read_text())  # the model's own feature list
    cols = ["cell_id", "date", *meta["feature_cols"]]  # IDs + the 52 features; fire_tomorrow left out on purpose
    df = pd.read_parquet(PROCESSED / "features.parquet", columns=cols)  # read only those columns
    df["date"] = pd.to_datetime(df["date"])  # dates as real datetimes
    start = df["date"].max() - pd.Timedelta(days=DAYS_KEPT - 1)  # first day to keep
    recent = df[df["date"] >= start].reset_index(drop=True)  # the recent window only
    recent.to_parquet(SERVING / "recent_features.parquet", index=False)  # save it
    return recent  # handy for the printout


if __name__ == "__main__":  # runs only when you call this file directly
    SERVING.mkdir(parents=True, exist_ok=True)  # create data/serving if it doesn't exist yet
    n_cells = export_grid()  # write the grid
    recent = export_features()  # write the features
    grid_kb = (SERVING / "grid.geojson").stat().st_size / 1e3  # grid file size in KB
    feat_mb = (SERVING / "recent_features.parquet").stat().st_size / 1e6  # features file size in MB
    full_mb = (PROCESSED / "features.parquet").stat().st_size / 1e6  # full training table size, for comparison
    print(f"grid.geojson: {n_cells} cells, {grid_kb:.0f} KB")  # grid summary
    print(f"recent_features.parquet: {len(recent):,} rows, "  # features summary: rows...
          f"{recent['date'].min():%Y-%m-%d} to {recent['date'].max():%Y-%m-%d}, {feat_mb:.1f} MB "  # ...date range and size...
          f"(full training table: {full_mb:.0f} MB)")  # ...against the full table