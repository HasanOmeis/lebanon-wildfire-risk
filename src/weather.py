from pathlib import Path  # Path builds file paths that work on any OS
import pandas as pd  # pandas handles the tables
import requests  # sends HTTP requests to the Open-Meteo API
import time  # lets the script pause between requests

CELLS_PATH = Path("data/processed/grid_points.csv")  # one lat/lon per 5 km cell (Day 3)
MAP_PATH = Path("data/processed/cell_to_weather_point.csv")  # which weather point each cell uses
POINTS_PATH = Path("data/processed/weather_points.csv")  # the unique weather points to download
STEP = 0.1  # weather grid spacing in degrees (~10 km, close to the model's own grid)
START_DATE = "2012-01-25"  # first fire date from step 1
END_DATE = "2026-10-29"  # last fire date from step 1
N_VARS = 7  # number of daily variables we will request
URL = "https://archive-api.open-meteo.com/v1/archive"  # Open-Meteo historical weather endpoint
DAILY_VARS = [  # the 7 daily variables we tested in step 2
    "temperature_2m_max",  # hottest temperature of the day (°C)
    "relative_humidity_2m_min",  # driest humidity of the day (%)
    "precipitation_sum",  # total rain that day (mm)
    "wind_speed_10m_max",  # strongest sustained wind (km/h)
    "wind_gusts_10m_max",  # strongest gust (km/h)
    "et0_fao_evapotranspiration",  # water the air pulls from plants and soil (mm)
    "vapour_pressure_deficit_max",  # how "thirsty" the air is at its driest (kPa)
]
RAW_DIR = Path("data/raw/weather")  # one Parquet file per downloaded batch goes here
GROUP_SIZE = 21  # weather points per request (126 / 21 = 6 groups)
DAILY_BUDGET = 9_000  # stop each run below the 10,000 daily limit, with a safety margin
HOURLY_BUDGET = 4_500  # pace requests below the 5,000 hourly limit


def snap_to_grid(cells, step=STEP):  # attach each cell to its nearest 0.1° weather point
    cells = cells.copy()  # work on a copy so the original table is untouched
    cells["wp_lat"] = (cells["lat"] / step).round() * step  # nearest multiple of 0.1 for latitude
    cells["wp_lon"] = (cells["lon"] / step).round() * step  # nearest multiple of 0.1 for longitude
    cells["wp_lat"] = cells["wp_lat"].round(1)  # clean float noise (e.g. 33.100000001 becomes 33.1)
    cells["wp_lon"] = cells["wp_lon"].round(1)  # same cleanup for longitude
    return cells  # give back cells with their weather point coordinates


def unique_points(cells):  # build the list of distinct weather points
    points = cells[["wp_lat", "wp_lon"]].drop_duplicates()  # keep each lat/lon pair once
    points = points.sort_values(["wp_lat", "wp_lon"]).reset_index(drop=True)  # stable order, south to north
    points["wp_id"] = points.index  # give each weather point an integer ID
    return points  # table of wp_id, wp_lat, wp_lon


def estimate_weight(n_points):  # estimate API cost with Open-Meteo's weighting formula
    n_days = (pd.Timestamp(END_DATE) - pd.Timestamp(START_DATE)).days + 1  # total days in the range
    return n_points * (n_days / 14) * (N_VARS / 10)  # locations × (days / 14) × (variables / 10)

def fetch_batch(points, start, end):  # download daily weather for several points over one date range
    params = {  # the query settings sent with the request
        "latitude": ",".join(points["wp_lat"].astype(str)),  # all latitudes as one string, e.g. "33.1,33.2"
        "longitude": ",".join(points["wp_lon"].astype(str)),  # all longitudes, same order as latitudes
        "start_date": start,  # first day of this batch
        "end_date": end,  # last day of this batch
        "daily": ",".join(DAILY_VARS),  # variables as one comma-separated string
        "timezone": "GMT",  # daily totals in UTC, same as FIRMS acq_date
    }
    response = requests.get(URL, params=params, timeout=120)  # send the request, give up after 120 s
    if response.status_code != 200:  # anything other than 200 is a failure
        raise RuntimeError(f"{response.status_code}: {response.text[:200]}")  # stop and show the API's reason
    data = response.json()  # convert the reply into Python objects
    if isinstance(data, dict):  # a single location comes back as one dict
        data = [data]  # wrap it in a list so the loop below works either way
    frames = []  # will hold one small table per weather point
    for wp_id, item in zip(points["wp_id"], data):  # results come back in the same order we asked
        df = pd.DataFrame(item["daily"])  # this point's days as a table
        df["wp_id"] = wp_id  # label every row with its weather point ID
        frames.append(df)  # add it to the list
    out = pd.concat(frames, ignore_index=True)  # stack all points into one table
    out = out.rename(columns={"time": "date"})  # clearer column name
    out["date"] = pd.to_datetime(out["date"])  # text dates become real dates
    return out  # one row per (weather point, day)

def batch_weight(n_points, start, end):  # API cost of one batch, using Open-Meteo's formula
    n_days = (pd.Timestamp(end) - pd.Timestamp(start)).days + 1  # days in this batch
    return n_points * (n_days / 14) * (N_VARS / 10)  # locations × (days / 14) × (variables / 10)


def make_batches(points):  # split the full job into (points, start, end, file) pieces
    groups = [points.iloc[i:i + GROUP_SIZE] for i in range(0, len(points), GROUP_SIZE)]  # 6 groups of 21 points
    first_year = int(START_DATE[:4])  # 2012
    last_year = int(END_DATE[:4])  # 2026
    batches = []  # will hold every batch
    for year in range(first_year, last_year + 1):  # one year at a time
        start = max(f"{year}-01-01", START_DATE)  # 2012 starts on Jan 25, not Jan 1
        end = min(f"{year}-12-31", END_DATE)  # 2026 ends on Oct 29, not Dec 31
        for g, group in enumerate(groups):  # every point group for this year
            path = RAW_DIR / f"weather_{year}_g{g}.parquet"  # e.g. weather_2019_g3.parquet
            batches.append((group, start, end, path))  # store everything needed to run this batch
    return batches  # 15 years × 6 groups = 90 batches


def fetch_with_retry(points, start, end, tries=3):  # call fetch_batch, retrying on failure
    for attempt in range(1, tries + 1):  # attempts 1, 2, 3
        try:  # try the download
            return fetch_batch(points, start, end)  # success: return the table right away
        except (RuntimeError, requests.RequestException) as err:  # API error or network problem
            print(f"  attempt {attempt} failed: {err}")  # show what went wrong
            if attempt == tries:  # this was the last attempt
                raise  # give up and stop the script
            time.sleep(60 * attempt)  # wait 1 min, then 2 min, before trying again


def download_all(points):  # download every missing batch, within today's budget
    RAW_DIR.mkdir(parents=True, exist_ok=True)  # create data/raw/weather if it doesn't exist
    todo = [b for b in make_batches(points) if not b[3].exists()]  # keep only batches with no saved file
    print(f"Batches left: {len(todo)}")  # how many still need downloading
    used = 0  # API weight spent in this run
    for i, (group, start, end, path) in enumerate(todo, 1):  # go through the missing batches
        w = batch_weight(len(group), start, end)  # cost of this batch
        if used + w > DAILY_BUDGET:  # this batch would go over today's budget
            print(f"Daily budget reached ({used:,.0f} used). Run again tomorrow.")  # explain why we stop
            return  # end the run; saved batches are kept
        df = fetch_with_retry(group, start, end)  # download the batch
        df.to_parquet(path, index=False)  # save it right away
        used += w  # add its cost to today's total
        print(f"[{i}/{len(todo)}] {path.name}: {len(df)} rows, weight {w:.0f}, used {used:,.0f}")  # progress line
        time.sleep(w / HOURLY_BUDGET * 3600)  # pause long enough to stay under the hourly limit
    print("All batches downloaded.")  # every batch file exists



if __name__ == "__main__":  # only run this block when the file is executed directly
    cells = pd.read_csv(CELLS_PATH)  # load the 488 cells
    cells = snap_to_grid(cells)  # attach the nearest weather point to each cell
    points = unique_points(cells)  # get the distinct points to download
    cells = cells.merge(points, on=["wp_lat", "wp_lon"])  # add wp_id to each cell
    cells[["cell_id", "wp_id"]].to_csv(MAP_PATH, index=False)  # save the cell to point lookup
    points.to_csv(POINTS_PATH, index=False)  # save the points list
    print("Cells:", len(cells))  # should still be 488
    print("Weather points:", len(points))  # expect roughly 100 to 130
    print("Cells per point (avg):", round(len(cells) / len(points), 1))  # about 4 cells share each point
    print(f"Estimated API weight: {estimate_weight(len(points)):,.0f} calls")  # total quota needed
    print(f"Days of free quota: {estimate_weight(len(points)) / 10_000:.1f}")  # at 10,000 calls per day
    batches = make_batches(points)  # build the batch list (no downloading)
    total = sum(batch_weight(len(g), s, e) for g, s, e, _ in batches)  # add up every batch's cost
    print("Batches:", len(batches))  # expect 90
    print(f"Total weight: {total:,.0f}")  # should match the ~33,970 estimate
    print("Example batch:", batches[0][1], "to", batches[0][2], "->", batches[0][3])  # first batch's dates and file
    download_all(points)  # download every missing batch, within today's budget

