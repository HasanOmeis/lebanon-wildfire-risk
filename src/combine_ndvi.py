from pathlib import Path  # clean file paths

import pandas as pd  # tables

RAW_DIR = Path("data/raw/ndvi")  # folder with one file per date
OUT_PATH = Path("data/processed/ndvi.parquet")  # the combined table we'll use from now on
EXPECTED_DATES = 663  # number of dates the search found
AVAILABILITY_LAG_DAYS = 3  # NASA publishes a composite ~2 days after it ends; 3 to be safe

files = sorted(RAW_DIR.glob("ndvi_*.parquet"))  # list every per-date file, in date order
print("Files found:", len(files), "of", EXPECTED_DATES)  # check 1: nothing missing

ndvi = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)  # stack all files into one table
ndvi = ndvi.sort_values(["cell_id", "start_date"]).reset_index(drop=True)  # order by cell, then time
ndvi["available_date"] = ndvi["end_date"] + pd.Timedelta(days=AVAILABILITY_LAG_DAYS)  # first day the model may use this value

span = (ndvi["end_date"] - ndvi["start_date"]).dt.days  # length of each composite in days (end minus start)
print("Composite length (days -> rows):", span.value_counts().sort_index().to_dict())  # check 2: mostly 15; the last one each year is shorter because composites restart on Jan 1

dates = pd.Series(sorted(ndvi["start_date"].unique()))  # unique start dates in order
gaps = dates.diff().dt.days.value_counts().sort_index().to_dict()  # days between one date and the next, counted
print("Gaps between dates (days -> count):", gaps)  # check 3: mostly 8; a few odd ones around New Year are normal

print("Rows with no NDVI (clouds/gaps):", f"{ndvi['ndvi'].isna().mean():.1%}")  # check 4: share of missing values
empty_cells = ndvi.groupby("cell_id")["ndvi"].apply(lambda s: s.isna().all())  # True for cells that never had a value
print("Cells never covered:", list(empty_cells[empty_cells].index))  # expect the 3 tiny sliver cells
print("Satellite rows:", ndvi["satellite"].value_counts().to_dict())  # Terra and Aqua should be about equal

OUT_PATH.parent.mkdir(parents=True, exist_ok=True)  # make sure data/processed exists
ndvi.to_parquet(OUT_PATH, index=False)  # save the combined table
print("Saved:", OUT_PATH, "| shape:", ndvi.shape)  # expect about 663 x 488 = 323,544 rows, 7 columns
print("Date range:", ndvi["start_date"].min().date(), "to", ndvi["start_date"].max().date())  # first and last composite