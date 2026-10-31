from pathlib import Path  # clean file paths that work on Windows and Linux

import geopandas as gpd  # read the grid and boundary files
import matplotlib.pyplot as plt  # quick visual check
import numpy as np  # array math
from odc.geo.geobox import GeoBox  # describes a pixel grid: size, position, projection (comes with odc-stac)
from rasterio.features import rasterize  # "paints" polygons onto a pixel grid (comes with rioxarray)

import odc.stac  # loads satellite images onto our pixel grid
import pandas as pd  # build and save tables
import planetary_computer  # signs download links
import pystac_client  # searches the catalog
from concurrent.futures import ThreadPoolExecutor, as_completed  # run several downloads at the same time


GRID_PATH = Path("data/processed/lebanon_grid.gpkg")  # Day 3 grid of ~5 km cells
BOUNDARY_PATH = Path("data/raw/lebanon_boundary.geojson")  # Lebanon outline
CRS = "EPSG:32636"  # UTM zone 36N, same as the grid, units in meters
RES = 250  # pixel size in meters, matches MODIS NDVI
LEBANON_BBOX = [35.1, 33.05, 36.65, 34.7]  # [min lon, min lat, max lon, max lat] for the catalog search
DATE_RANGE = "2011-12-01/2026-10-31"  # starts a bit before 2012 so early January is covered
NDVI_BAND = "250m_16_days_NDVI"  # the layer we want from each image
OUT_DIR = Path("data/raw/ndvi")  # one small parquet file per date goes here
STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"  # catalog address
WORKERS = 6  # how many dates download in parallel (polite to the server, much faster than 1)




def build_cell_raster():  # build the pixel grid and the cell-ID image once, reused for every NDVI date
    grid = gpd.read_file(GRID_PATH)  # load the 488 grid cells
    boundary = gpd.read_file(BOUNDARY_PATH).to_crs(grid.crs)  # load the outline and convert it to the grid's projection
    land_cells = gpd.clip(grid, boundary)  # cut each cell to the part inside Lebanon (drops sea and Syria pieces)

    geobox = GeoBox.from_bbox(tuple(grid.total_bounds), crs=CRS, resolution=RES)  # fixed 250 m pixel grid covering the whole cell grid

    cell_raster = rasterize(  # paint each pixel with the ID of the cell it falls in
        shapes=zip(land_cells.geometry, land_cells["cell_id"]),  # pairs of (shape, value to paint)
        out_shape=tuple(geobox.shape),  # number of rows and columns of the pixel grid
        transform=geobox.transform,  # where the pixel grid sits on the map
        fill=-1,  # pixels outside every cell get -1
        dtype="int32",  # whole numbers are enough for cell IDs
    )  # end of rasterize
    return geobox, cell_raster  # hand back both for the NDVI loop

def search_items():  # find all NDVI images and group them by date
    catalog = pystac_client.Client.open(STAC_URL)  # connect without signing; we sign right before each download so links never expire mid-run
    search = catalog.search(collections=["modis-13Q1-061"], bbox=LEBANON_BBOX, datetime=DATE_RANGE)  # all images touching Lebanon

    latest = {}  # will hold one item per (date, tile)
    for item in search.items():  # go through every result
        key = (item.properties["start_datetime"][:10], item.properties["modis:horizontal-tile"])  # identify by date + tile
        if key not in latest or item.id > latest[key].id:  # IDs end with a processing timestamp, so the bigger ID is the newer reprocessed version
            latest[key] = item  # keep only the newest copy (fixes the 2024-05-08 duplicates)

    by_date = {}  # date -> list of tiles for that date
    for (date, _tile), item in latest.items():  # regroup by date only
        by_date.setdefault(date, []).append(item)  # add this tile to its date's list
    return dict(sorted(by_date.items()))  # return dates in time order


def cell_means(items, geobox, cell_raster, n_cells):  # average NDVI per cell for one date
    ds = odc.stac.load(items, bands=[NDVI_BAND], geobox=geobox, groupby="solar_day", nodata=-3000)  # download + stitch tiles onto our exact pixel grid
    raw = ds[NDVI_BAND].isel(time=0).values  # plain numpy 2D array of raw values

    valid = (raw >= -2000) & (raw <= 10000) & (cell_raster >= 0)  # good NDVI pixels that are inside a Lebanon cell
    ids = cell_raster[valid]  # the cell ID of each good pixel
    vals = raw[valid] * 0.0001  # the scaled NDVI value of each good pixel

    sums = np.bincount(ids, weights=vals, minlength=n_cells)  # total NDVI per cell (fast groupby-sum)
    counts = np.bincount(ids, minlength=n_cells)  # number of good pixels per cell
    totals = np.bincount(cell_raster[cell_raster >= 0], minlength=n_cells)  # number of pixels per cell, good or not

    mean = np.where(counts > 0, sums / np.maximum(counts, 1), np.nan)  # average; NaN where a cell has no good pixels
    valid_frac = counts / np.maximum(totals, 1)  # share of the cell that had good data (low = clouds/gaps)
    return pd.DataFrame({"cell_id": np.arange(n_cells), "ndvi": mean, "valid_frac": valid_frac})  # one row per cell

def process_date(start_date, items, geobox, cell_raster, n_cells):  # do all the work for one date and save it
    out_file = OUT_DIR / f"ndvi_{start_date}.parquet"  # file name for this date
    signed = [planetary_computer.sign(it) for it in items]  # fresh signed links for this date
    df = cell_means(signed, geobox, cell_raster, n_cells)  # per-cell NDVI table
    df["start_date"] = pd.to_datetime(start_date)  # first day of the 16-day composite
    df["end_date"] = pd.to_datetime(items[0].properties["end_datetime"][:10])  # last day: the earliest the value could be known
    df["satellite"] = items[0].id[:3]  # MOD = Terra, MYD = Aqua
    df.to_parquet(out_file, index=False)  # save this date
    return df["ndvi"].mean()  # send back the mean for the progress line


if __name__ == "__main__":  # runs only when we run this file directly
    OUT_DIR.mkdir(parents=True, exist_ok=True)  # create data/raw/ndvi if it doesn't exist
    geobox, cell_raster = build_cell_raster()  # fixed pixel grid + cell IDs, built once
    n_cells = len(gpd.read_file(GRID_PATH))  # 488 cells, IDs 0 to 487

    by_date = search_items()  # all dates with their tiles
    print("Dates to process:", len(by_date))  # should be about 659

    todo = {d: its for d, its in by_date.items() if not (OUT_DIR / f"ndvi_{d}.parquet").exists()}  # only dates not saved yet
    print("Dates left:", len(todo))  # should be about 663 - 108 = 555

    with ThreadPoolExecutor(max_workers=WORKERS) as pool:  # open a pool of 6 worker threads
        futures = {pool.submit(process_date, d, its, geobox, cell_raster, n_cells): d for d, its in todo.items()}  # queue every date; remember which future is which date
        for n, future in enumerate(as_completed(futures), start=1):  # handle each date as soon as it finishes, in any order
            d = futures[future]  # which date this was
            try:  # one failed date shouldn't stop the others
                mean = future.result()  # get the mean NDVI (or raise the error that happened)
                print(f"[{n}/{len(todo)}] {d} ok, mean NDVI {mean:.3f}")  # progress line
            except Exception as e:  # something went wrong for this date
                print(f"[{n}/{len(todo)}] {d} FAILED: {e}")  # report it; rerun the script later to retry