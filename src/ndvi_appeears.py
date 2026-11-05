import re  # reads the satellite and date from each file name
from pathlib import Path  # clean file paths

import geopandas as gpd  # reads the grid to count the cells
import numpy as np  # array math
import pandas as pd  # tables
import rasterio  # opens the GeoTIFF files
from rasterio.warp import reproject, Resampling  # moves pixels onto our UTM grid

from ndvi import build_cell_raster, GRID_PATH, OUT_DIR  # same pixel grid, cell IDs and output folder as Day 5

IN_DIR = Path("data/raw/appeears")  # the 23 files from NASA AppEEARS
NODATA = -3000  # MODIS "no data" value
NAME_RE = re.compile(r"^(MOD|MYD)13Q1\.061__250m_16_days_NDVI_(\d{8})T")  # e.g. MOD13Q1.061__250m_16_days_NDVI_20250610T...


def load_on_grid(path, geobox):  # read one GeoTIFF and place it on our fixed 250 m UTM grid
    out = np.full(tuple(geobox.shape), NODATA, dtype="float32")  # empty image the size of our grid, filled with "no data"
    with rasterio.open(path) as src:  # open the NASA file
        reproject(  # copy every pixel to where it belongs on our grid
            source=rasterio.band(src, 1),  # the NDVI layer
            destination=out,  # our empty image
            src_transform=src.transform,  # where the NASA pixels sit on the map
            src_crs=src.crs,  # NASA file is in lat/lon (Geographic)
            dst_transform=geobox.transform,  # where our pixels sit
            dst_crs="EPSG:32636",  # our grid is in UTM 36N meters
            src_nodata=src.nodata if src.nodata is not None else NODATA,  # NASA's "no data" value
            dst_nodata=NODATA,  # keep "no data" as -3000 in our image
            resampling=Resampling.nearest,  # take the nearest pixel, no blending (same idea as Day 5)
        )  # end of reproject
    return out  # NDVI image on our grid, raw values


def cell_means(raw, cell_raster, n_cells):  # average NDVI per cell (same rules as ndvi.py)
    if np.nanmax(raw) <= 1.5:  # if NASA already scaled the values to -0.2..1.0
        raw = np.where(raw == NODATA, NODATA, raw * 10000)  # put them back on the raw scale so one rule fits both
    valid = (raw >= -2000) & (raw <= 10000) & (cell_raster >= 0)  # good NDVI pixels inside a Lebanon cell
    ids = cell_raster[valid]  # the cell ID of each good pixel
    vals = raw[valid] * 0.0001  # scaled NDVI of each good pixel
    sums = np.bincount(ids, weights=vals, minlength=n_cells)  # total NDVI per cell
    counts = np.bincount(ids, minlength=n_cells)  # good pixels per cell
    totals = np.bincount(cell_raster[cell_raster >= 0], minlength=n_cells)  # all pixels per cell
    mean = np.where(counts > 0, sums / np.maximum(counts, 1), np.nan)  # average; NaN where no good pixels
    return pd.DataFrame({"cell_id": np.arange(n_cells), "ndvi": mean, "valid_frac": counts / np.maximum(totals, 1)})  # one row per cell


if __name__ == "__main__":  # runs only when you call this file directly
    geobox, cell_raster = build_cell_raster()  # same pixel grid and cell IDs as Day 5
    n_cells = len(gpd.read_file(GRID_PATH))  # 488 cells
    added, skipped = 0, 0  # counters for the summary
    for path in sorted(IN_DIR.glob("*.tif")):  # every NASA file, in name order
        m = NAME_RE.match(path.name)  # pull satellite and date from the name
        if m is None:  # not an NDVI file we expect
            print("Skipping unknown file:", path.name)  # report it
            continue  # move on
        satellite, ymd = m.group(1), m.group(2)  # "MOD"/"MYD" and "20250610"
        start = pd.to_datetime(ymd, format="%Y%m%d")  # composite start date
        out_file = OUT_DIR / f"ndvi_{start.date()}.parquet"  # same file name pattern as ndvi.py
        if out_file.exists():  # Day 5 already has this date from the online catalog
            skipped += 1  # count it
            continue  # never overwrite existing data
        raw = load_on_grid(path, geobox)  # NASA image on our grid
        df = cell_means(raw, cell_raster, n_cells)  # per-cell NDVI
        df["start_date"] = start  # first day of the 16-day composite
        df["end_date"] = start + pd.Timedelta(days=15)  # last day of the composite (16 days in total)
        df["satellite"] = satellite  # MOD = Terra, MYD = Aqua
        df.to_parquet(out_file, index=False)  # save in the same format as Day 5
        added += 1  # count it
        print(f"{start.date()} {satellite} added, mean NDVI {df['ndvi'].mean():.3f}, valid {df['valid_frac'].mean():.0%}")  # progress line
    print(f"Added {added} new dates, skipped {skipped} that already existed")  # summary