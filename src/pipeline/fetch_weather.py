"""Daily weather update: recent days from Open-Meteo's forecast API, copied from 126 points to 488 cells."""  # file description

import datetime as dt  # dates and day steps
import logging  # log lines for the runner
import time  # pauses between requests and retries

import pandas as pd  # tables
import requests  # HTTP requests to Open-Meteo

from . import config as C  # pipeline settings (Step 2)

log = logging.getLogger("pipeline.weather")  # this step's own logger

URL = "https://api.open-meteo.com/v1/forecast"  # forecast API: covers recent past days and today
DAILY_VARS = [  # the same 7 variables as Day 4, same names as the file's columns
    "temperature_2m_max",  # hottest temperature of the day (°C)
    "relative_humidity_2m_min",  # driest humidity of the day (%)
    "precipitation_sum",  # total rain that day (mm)
    "wind_speed_10m_max",  # strongest sustained wind (km/h)
    "wind_gusts_10m_max",  # strongest gust (km/h)
    "et0_fao_evapotranspiration",  # water the air pulls from plants and soil (mm)
    "vapour_pressure_deficit_max",  # how "thirsty" the air is at its driest (kPa)
]  # end of the variable list
BATCH = 25  # weather points per request (keeps the URL short)
MAX_BACK_DAYS = 90  # the forecast API only reaches about 3 months back


def fetch_points(points: pd.DataFrame, start: dt.date, end: dt.date) -> pd.DataFrame:  # one request for several points
    params = {  # query settings
        "latitude": ",".join(points["wp_lat"].astype(str)),  # all latitudes, comma-separated
        "longitude": ",".join(points["wp_lon"].astype(str)),  # all longitudes, same order
        "daily": ",".join(DAILY_VARS),  # the 7 variables
        "timezone": "GMT",  # daily values in UTC, same as Day 4 and FIRMS
        "start_date": start.isoformat(),  # first day
        "end_date": end.isoformat(),  # last day
    }  # end of the settings
    for attempt in range(1, 4):  # up to 3 tries
        resp = requests.get(URL, params=params, timeout=60)  # send the request
        if resp.status_code == 200:  # success
            break  # stop retrying
        if resp.status_code in (429, 500, 502, 503) and attempt < 3:  # rate limit or server trouble
            log.warning("HTTP %d, retry %d in %d s", resp.status_code, attempt, 20 * attempt)  # record the retry
            time.sleep(20 * attempt)  # wait longer each time
            continue  # try again
        raise RuntimeError(f"Open-Meteo error {resp.status_code}: {resp.text[:200]}")  # give up with the reason
    data = resp.json()  # reply as Python objects
    data = data if isinstance(data, list) else [data]  # one point returns a dict, several return a list
    frames = []  # one table per point
    for wp_id, item in zip(points["wp_id"], data):  # replies come back in the same order as the points
        df = pd.DataFrame(item["daily"])  # "time" column + the 7 variables
        df["wp_id"] = wp_id  # remember which point this is
        frames.append(df)  # keep it
    return pd.concat(frames, ignore_index=True)  # all points in one table


def update_weather(end: dt.date | None = None) -> int:  # main step: returns how many new cell-days were added
    end = end or C.today_utc()  # default: up to today (today's values come from the forecast)
    existing = pd.read_parquet(C.WEATHER)  # all weather so far
    if "source" not in existing.columns:  # first run: Day 4 rows have no tag yet
        existing["source"] = "archive"  # everything until now came from the archive API
    last_archive = existing.loc[existing["source"] == "archive", "date"].max().date()  # newest final day
    start = min(last_archive + dt.timedelta(days=1), end - dt.timedelta(days=C.WEATHER_PAST_DAYS - 1))  # catch up, or re-check 7 days
    start = max(start, end - dt.timedelta(days=MAX_BACK_DAYS - 1))  # stay inside the API's reach
    log.info("Fetching weather %s to %s (archive ends %s)", start, end, last_archive)  # record the window

    points = pd.read_csv(C.WEATHER_POINTS)  # 126 points: wp_lat, wp_lon, wp_id
    frames = []  # one table per batch
    for i in range(0, len(points), BATCH):  # walk through the points 25 at a time
        frames.append(fetch_points(points.iloc[i:i + BATCH], start, end))  # fetch this batch
        time.sleep(1)  # be polite to the API
    raw = pd.concat(frames, ignore_index=True).rename(columns={"time": "date"})  # all points, "time" renamed to "date"
    raw["date"] = pd.to_datetime(raw["date"])  # text -> real dates

    links = pd.read_csv(C.CELL_TO_POINT)  # cell_id -> wp_id
    new = links.merge(raw, on="wp_id", how="left").drop(columns="wp_id")  # copy each point's weather to its cells
    missing = new[DAILY_VARS].isna().any(axis=1)  # rows where the API returned an empty value
    if missing.any():  # report and drop them
        log.warning("Dropped %d cell-days with missing weather", int(missing.sum()))  # how many
        new = new[~missing]  # keep complete rows only
    new["relative_humidity_2m_min"] = new["relative_humidity_2m_min"].round()  # whole numbers, like the old file
    new["source"] = "forecast"  # tag the new rows
    new = new[existing.columns].astype(existing.dtypes.to_dict())  # same columns, order and types as the old file

    keep = ~((existing["source"] == "forecast") & (existing["date"] >= new["date"].min()))  # old forecast rows in this window get replaced
    old = existing[keep]  # everything we keep from before (all archive rows + older forecasts)
    old_keys = pd.MultiIndex.from_frame(old[["cell_id", "date"]])  # (cell, date) pairs we already have
    new = new[~pd.MultiIndex.from_frame(new[["cell_id", "date"]]).isin(old_keys)]  # never overwrite an archive row
    combined = pd.concat([old, new], ignore_index=True).sort_values(["cell_id", "date"]).reset_index(drop=True)  # merge and order

    per_day = new.groupby("date").size()  # cells per new day
    bad = per_day[per_day != C.EXPECTED_CELLS]  # days that aren't complete
    if not bad.empty:  # warn, the Step 7 checks will decide if that's fatal
        log.warning("Incomplete days: %s", bad.to_dict())  # which days and how many cells

    tmp = C.WEATHER.with_suffix(".tmp.parquet")  # temp file next to the real one
    combined.to_parquet(tmp, index=False)  # write it fully first
    tmp.replace(C.WEATHER)  # then swap it in
    added = len(combined) - len(existing)  # net growth of the file
    log.info("Weather rows: %d (+%d); forecast rows: %d; last date %s",  # summary line...
             len(combined), added, int((combined["source"] == "forecast").sum()), combined["date"].max().date())  # ...values
    return added  # used by the runner's report


if __name__ == "__main__":  # run alone: python -m src.pipeline.fetch_weather
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(name)s | %(message)s")  # show log lines
    update_weather()  # run the update up to today