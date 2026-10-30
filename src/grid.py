from pathlib import Path  # Path builds file paths that work on any OS
import geopandas as gpd  # geopandas reads and works with map data (shapes + coordinates)
import numpy as np  # numpy creates evenly spaced number ranges for the grid edges
from shapely.geometry import box  # box() makes a rectangle shape from (minx, miny, maxx, maxy)
import matplotlib.pyplot as plt  # matplotlib draws charts and maps


BOUNDARY_PATH = Path("data/raw/lebanon_boundary.geojson")  # the border file we just downloaded
METRIC_CRS = "EPSG:32636"  # UTM zone 36N, a projection in meters that covers Lebanon
CELL_SIZE = 5000  # grid cell width/height in meters (5 km)
PROCESSED_DIR = Path("data/processed")  # folder for cleaned/derived data
GRID_PATH = PROCESSED_DIR / "lebanon_grid.gpkg"  # cell shapes (GeoPackage keeps geometry + CRS)
POINTS_PATH = PROCESSED_DIR / "grid_points.csv"  # one lat/lon per cell, used for weather on Day 4
FIG_PATH = Path("reports/figures/grid_check.png")  # where the check map image will be saved


def load_boundary():  # function that loads Lebanon's border
    boundary = gpd.read_file(BOUNDARY_PATH)  # read the GeoJSON into a GeoDataFrame
    return boundary  # give the GeoDataFrame back to whoever called this
def to_metric(gdf):  # function that converts any GeoDataFrame to meters
    return gdf.to_crs(METRIC_CRS)  # reproject from lat/lon degrees to UTM meters
def make_grid(bounds, cell_size=CELL_SIZE):  # build square cells covering a bounding box
    min_x, min_y, max_x, max_y = bounds  # unpack the box edges (in meters)
    xs = np.arange(min_x, max_x, cell_size)  # left edge of every column, stepping 5 km
    ys = np.arange(min_y, max_y, cell_size)  # bottom edge of every row, stepping 5 km
    cells = [box(x, y, x + cell_size, y + cell_size) for x in xs for y in ys]  # one 5x5 km square per (column, row)
    grid = gpd.GeoDataFrame(geometry=cells, crs=METRIC_CRS)  # wrap the squares in a GeoDataFrame in meters
    return grid  # give the full rectangular grid back
def clip_to_country(grid, country):  # keep only cells that touch the country
    shape = country.geometry.iloc[0]  # the single Lebanon polygon (in meters)
    grid = grid[grid.intersects(shape)].copy()  # drop cells fully in the sea or Syria
    grid["land_frac"] = grid.intersection(shape).area / grid.area  # share of each cell that is inside Lebanon (0 to 1)
    grid = grid.reset_index(drop=True)  # renumber rows 0, 1, 2... after filtering
    grid["cell_id"] = grid.index  # give each cell a stable integer ID
    return grid[["cell_id", "land_frac", "geometry"]]  # return only the columns we need, in a clean order
def add_points(grid, country):  # add a lat/lon point for each cell, placed on land
    shape = country.geometry.iloc[0]  # the single Lebanon polygon (in meters)
    land = grid.intersection(shape)  # the part of each cell that is inside Lebanon
    pts = gpd.GeoSeries(land.representative_point(), crs=METRIC_CRS)  # one point guaranteed to sit inside that land part
    pts_ll = pts.to_crs("EPSG:4326")  # convert the points from meters back to lat/lon degrees
    grid["lon"] = pts_ll.x.round(4)  # longitude, rounded to 4 decimals (about 10 m)
    grid["lat"] = pts_ll.y.round(4)  # latitude, rounded to 4 decimals (about 10 m)
    return grid  # give the grid back with lat/lon columns added
def save_grid(grid):  # write the grid to disk in two formats
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)  # create data/processed if it doesn't exist yet
    grid.to_file(GRID_PATH, driver="GPKG")  # save shapes + attributes as a GeoPackage
    grid[["cell_id", "lat", "lon", "land_frac"]].to_csv(POINTS_PATH, index=False)  # save a plain table without geometry
    print(f"Saved {GRID_PATH} and {POINTS_PATH}")  # confirm both files were written
def plot_grid(grid, country):  # draw the grid on top of Lebanon's border
    FIG_PATH.parent.mkdir(parents=True, exist_ok=True)  # create reports/figures if it doesn't exist
    fig, ax = plt.subplots(figsize=(6, 9))  # tall figure, because Lebanon is taller than it is wide
    grid.plot(ax=ax, column="land_frac", cmap="YlGn", edgecolor="grey", linewidth=0.3, legend=True)  # cells colored by land share
    country.boundary.plot(ax=ax, color="red", linewidth=1)  # Lebanon's border as a red outline
    ax.set_title(f"Lebanon 5 km grid: {len(grid)} cells")  # title with the cell count
    ax.set_axis_off()  # hide the meter-based axis numbers
    fig.savefig(FIG_PATH, dpi=150, bbox_inches="tight")  # save the image at good quality
    print(f"Saved {FIG_PATH}")  # confirm the image was written
    plt.show()  # open a window showing the map


if __name__ == "__main__":  # only run this block when the file is executed directly
    lebanon = load_boundary()  # load the border (in lat/lon degrees)
    lebanon_m = to_metric(lebanon)  # convert the border to meters
    print("CRS:", lebanon_m.crs)  # should now show EPSG:32636
    print("Bounds (m):", lebanon_m.total_bounds)  # [min_x, min_y, max_x, max_y] in meters
    area_km2 = lebanon_m.area.iloc[0] / 1_000_000  # area in m² divided by 1,000,000 gives km²
    print(f"Area: {area_km2:,.0f} km²")  # Lebanon's real area is about 10,452 km²
    grid = make_grid(lebanon_m.total_bounds)  # build a grid over Lebanon's whole bounding box
    print("Cells in bounding box:", len(grid))  # count of all squares (includes sea + Syria for now)
    print("One cell area (km²):", grid.area.iloc[0] / 1_000_000)  # should be exactly 25.0
    grid = clip_to_country(grid, lebanon_m)  # keep only cells touching Lebanon
    print("Cells touching Lebanon:", len(grid))  # number of cells we'll predict for
    print("Fully inside cells:", (grid["land_frac"] > 0.99).sum())  # cells that are entirely land in Lebanon
    print(grid["land_frac"].describe())  # summary of how much of each cell is inside Lebanon
    grid = add_points(grid, lebanon_m)  # add lat/lon for each cell
    print(grid[["cell_id", "land_frac", "lat", "lon"]].head())  # preview the first cells
    print("Lat range:", grid["lat"].min(), "to", grid["lat"].max())  # should be about 33.05 to 34.69
    print("Lon range:", grid["lon"].min(), "to", grid["lon"].max())  # should be about 35.1 to 36.6
    save_grid(grid)  # write the finished grid to data/processed
    plot_grid(grid, lebanon_m)  # draw and save the check ma