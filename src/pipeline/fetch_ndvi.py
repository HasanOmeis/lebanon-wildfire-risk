"""Daily NDVI update: download any new MODIS composites with the Day 5 functions and append them."""  # file description

import datetime as dt  # dates and day steps
import logging  # log lines for the runner

import geopandas as gpd  # counts the grid cells
import pandas as pd  # tables

from src import ndvi as N  # Day 5 code: search_items, build_cell_raster, process_date, OUT_DIR, GRID_PATH

from . import config as C  # pipeline settings (Step 2)

log = logging.getLogger("pipeline.ndvi")  # this step's own logger
SEARCH_BACK_DAYS = 20  # look a bit before the newest composite, so a late-published one isn't missed


def update_ndvi(end: dt.date | None = None) -> int:  # main step: returns how many new composites were added
    end = end or C.today_utc()  # default: search up to today
    existing = pd.read_parquet(C.NDVI)  # all NDVI so far
    newest = existing["start_date"].max().date()  # start of the newest composite we have
    start = newest - dt.timedelta(days=SEARCH_BACK_DAYS)  # search window start
    N.DATE_RANGE = f"{start}/{end}"  # point Day 5's search at recent weeks only (it reads this setting when called)
    by_date = N.search_items()  # {start date text: [tiles]} for the window
    have = set(existing["start_date"].dt.strftime("%Y-%m-%d"))  # composite dates already in the file
    todo = {d: items for d, items in by_date.items() if d not in have}  # only composites we don't have
    log.info("Catalog %s to %s: %d composites, %d new", start, end, len(by_date), len(todo))  # record the search

    if not todo:  # nothing new published yet
        log.info("No new NDVI; features will forward-fill and ndvi_age_days will grow")  # normal between composites
        return 0  # nothing added

    N.OUT_DIR.mkdir(parents=True, exist_ok=True)  # data/raw/ndvi, in case it was cleaned
    geobox, cell_raster = N.build_cell_raster()  # same pixel grid and cell IDs as Day 5
    n_cells = len(gpd.read_file(N.GRID_PATH))  # 488
    for d, items in sorted(todo.items()):  # oldest first
        mean = N.process_date(d, items, geobox, cell_raster, n_cells)  # Day 5 math; saves data/raw/ndvi/ndvi_<d>.parquet
        log.info("Composite %s done, mean NDVI %.3f", d, mean)  # progress line

    new = pd.concat([pd.read_parquet(N.OUT_DIR / f"ndvi_{d}.parquet") for d in todo], ignore_index=True)  # the new per-date files
    lag = (existing["available_date"] - existing["end_date"]).mode().iloc[0]  # the file's own availability rule (3 days today)
    new["available_date"] = new["end_date"] + lag  # same leakage-safe date as the history
    new = new[existing.columns].astype(existing.dtypes.to_dict())  # same columns, order and types

    combined = pd.concat([existing, new], ignore_index=True)  # old + new
    combined = combined.drop_duplicates(subset=["cell_id", "start_date", "satellite"], keep="first")  # never twice
    combined = combined.sort_values(["end_date", "satellite", "cell_id"]).reset_index(drop=True)  # same order as merge_ndvi.py
    tmp = C.NDVI.with_suffix(".tmp.parquet")  # temp file next to the real one
    combined.to_parquet(tmp, index=False)  # write it fully first
    tmp.replace(C.NDVI)  # then swap it in
    log.info("NDVI rows: %d (+%d); newest available %s", len(combined), len(combined) - len(existing),  # summary...
             combined["available_date"].max().date())  # ...newest date the model may use
    return len(todo)  # composites added


def check_terrain() -> None:  # terrain never changes; just make sure it's intact
    n = pd.read_parquet(C.TERRAIN)["cell_id"].nunique()  # cells with terrain
    if n != C.EXPECTED_CELLS:  # should be 488
        raise RuntimeError(f"terrain.parquet has {n} cells, expected {C.EXPECTED_CELLS}")  # stop the run
    log.info("Terrain OK: %d cells (static, not fetched)", n)  # record the check


if __name__ == "__main__":  # run alone: python -m src.pipeline.fetch_ndvi
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(name)s | %(message)s")  # show log lines
    update_ndvi()  # add any new composites
    check_terrain()  # confirm terrain is fine