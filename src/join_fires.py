from pathlib import Path  # Path builds file paths that work on any OS
import pandas as pd  # pandas loads the fire table
import geopandas as gpd  # geopandas handles points, cells and spatial joins

FIRES_PATH = Path("data/raw/firms_lebanon.parquet")  # cleaned fire detections from Day 2
GRID_PATH = Path("data/processed/lebanon_grid.gpkg")  # the 5 km grid we saved in step 7
OUT_PATH = Path("data/processed/fires_with_cells.parquet")  # where fires + cell_id will be saved
METRIC_CRS = "EPSG:32636"  # same meter-based projection as the grid


def load_fire_points():  # turn the fire table into map points
    df = pd.read_parquet(FIRES_PATH)  # load the fire detections
    points = gpd.points_from_xy(df["longitude"], df["latitude"])  # make a point from each lon/lat pair
    fires = gpd.GeoDataFrame(df, geometry=points, crs="EPSG:4326")  # wrap as a GeoDataFrame in lat/lon
    return fires.to_crs(METRIC_CRS)  # convert to meters so it matches the grid


def join_to_grid(fires, grid):  # attach a cell_id to every fire
    joined = gpd.sjoin(fires, grid[["cell_id", "geometry"]], how="left", predicate="within")  # find the cell each fire falls inside
    outside = joined["cell_id"].isna().sum()  # fires that landed in no cell (sea or Syria)
    print(f"Fires outside the grid (dropped): {outside}")  # report how many were dropped
    joined = joined.dropna(subset=["cell_id"])  # keep only fires inside a Lebanon cell
    joined["cell_id"] = joined["cell_id"].astype(int)  # cell_id becomes float after the join, so convert back to int
    return joined.drop(columns=["index_right", "geometry"])  # remove join helper column and geometry to get a plain table
def fire_summary(fires, grid):  # print checks on how fires spread over the cells
    counts = fires.groupby("cell_id").size().rename("n_fires")  # number of detections in each cell
    print(f"Cells with at least 1 fire: {len(counts)} of {len(grid)}")  # how much of Lebanon has ever burned
    top = counts.sort_values(ascending=False).head(10).reset_index()  # the 10 cells with the most detections
    top = top.merge(grid[["cell_id", "lat", "lon", "land_frac"]], on="cell_id")  # add each top cell's location
    print("\nTop 10 cells by fire detections:")  # header for the table
    print(top)  # show cell_id, n_fires, lat, lon, land_frac
    fire_days = fires.groupby(["cell_id", "acq_date"]).ngroups  # unique (cell, day) pairs with a fire
    total = len(grid) * fires["acq_date"].nunique()  # rough count of cell-days, using only days with any fire
    print(f"\nCell-days with fire: {fire_days} (~{fire_days / total:.2%} of cell-days on fire days)")  # shows how rare fire is


def flag_hotspots(fires, min_years=6, min_months=8):  # flag spots active in many years AND many calendar months
    spot = fires["latitude"].round(2).astype(str) + "_" + fires["longitude"].round(2).astype(str)  # ~1 km spot ID from rounded lat/lon
    fires = fires.assign(spot=spot, year=fires["acq_date"].dt.year, month=fires["acq_date"].dt.month)  # add spot ID, year and month columns
    years_per_spot = fires.groupby("spot")["year"].nunique()  # how many different years each spot had a detection
    months_per_spot = fires.groupby("spot")["month"].nunique()  # how many different calendar months (1-12) each spot was active in
    hot_spots = years_per_spot[(years_per_spot >= min_years) & (months_per_spot >= min_months)].index  # must be both long-lived AND year-round
    fires["hotspot"] = fires["spot"].isin(hot_spots)  # True if the detection is at a persistent spot
    print(f"\nPersistent spots (>= {min_years} years, >= {min_months} months): {len(hot_spots)}")  # how many fixed heat sources were found
    print(f"Detections flagged: {fires['hotspot'].sum()} of {len(fires)}")  # how many fires those spots account for
    print("Flagged detections per cell (top 10):")  # header for the next table
    print(fires[fires["hotspot"]].groupby("cell_id").size().sort_values(ascending=False).head(10))  # which cells hold the hotspots
    return fires.drop(columns=["spot", "year", "month"])  # remove helper columns, keep the hotspot flag


if __name__ == "__main__":  # only run this block when the file is executed directly
    grid = gpd.read_file(GRID_PATH)  # load the saved grid
    fires = load_fire_points()  # load fires as points in meters
    print(f"Fires loaded: {len(fires)}")  # total detections before the join
    fires = join_to_grid(fires, grid)  # attach cell_id and drop outside points
    print(f"Fires inside Lebanon cells: {len(fires)}")  # detections we'll actually use
    fire_summary(fires, grid)  # print the sanity checks
    fires = flag_hotspots(fires)  # flag persistent industrial-looking spots
    fires.to_parquet(OUT_PATH, index=False)  # save fires with cell_id and hotspot flag
    print(f"Saved {OUT_PATH} with columns: {list(fires.columns)}")  # confirm the save and show the columns