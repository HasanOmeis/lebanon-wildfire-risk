"""Daily fire update: download recent FIRMS NRT detections and append the new ones."""  # file description

import datetime as dt  # dates and day steps
import logging  # log lines instead of print, so the runner can save them
import time  # short pause between requests
from io import StringIO  # lets pandas read CSV text from the API

import pandas as pd  # tables
import requests  # HTTP requests to FIRMS

from . import config as C  # pipeline settings (Step 2), short name C

log = logging.getLogger("pipeline.fires")  # this step's own logger
MAX_CATCHUP_DAYS = 60  # never reach further back than this (NRT only covers recent months)


def download(start: dt.date, days: int) -> pd.DataFrame:  # fetch 'days' days (max 5) starting on 'start'
    url = f"{C.FIRMS_URL}/{C.firms_key()}/{C.FIRMS_SOURCE}/{C.BBOX}/{days}/{start.isoformat()}"  # key/source/box/days/start
    resp = requests.get(url, timeout=60)  # send the request, give up after 60 s
    resp.raise_for_status()  # stop on HTTP errors (500, 503, ...)
    text = resp.text.strip()  # the CSV text
    if not text.startswith("latitude"):  # a real CSV starts with its header; anything else is an error message
        raise RuntimeError(f"FIRMS did not return CSV: {text[:120]}")  # show the start of the message (never the key)
    df = pd.read_csv(StringIO(text))  # parse the CSV (may have only the header = no fires)
    end = start + dt.timedelta(days=days - 1)  # last day of this chunk, for the log line
    log.info("Downloaded %d detections for %s to %s", len(df), start, end)  # record what came back
    return df  # raw detections


def clean(df: pd.DataFrame, existing: pd.DataFrame) -> pd.DataFrame:  # same rules as src/clean_firms.py
    if df.empty:  # no fires in the window
        return existing.iloc[0:0]  # empty table with the right columns
    df["confidence"] = df["confidence"].astype(str).str.lower().str[0]  # 'nominal' / 'n' -> 'n', etc.
    df = df[df["confidence"] != "l"].copy()  # drop low-confidence detections, like Day 2
    df["acq_date"] = pd.to_datetime(df["acq_date"])  # text -> real dates
    nrt = existing.loc[existing["source"].astype(str).str.contains("NRT"), "source"]  # how Day 2 labelled NRT rows
    df["source"] = nrt.iloc[0] if len(nrt) else C.FIRMS_SOURCE  # reuse that exact label
    df = df.reindex(columns=existing.columns)  # same columns, same order as the old file
    return df.astype(existing.dtypes.to_dict())  # same types as the old file


def fetch_window(start: dt.date, end: dt.date) -> pd.DataFrame:  # download start..end in 5-day chunks
    chunks = []  # one table per chunk
    day = start  # first chunk starts here
    while day <= end:  # until we pass the last day
        days = min(C.FIRMS_DAYS, (end - day).days + 1)  # 5 days, or fewer for the last chunk
        chunks.append(download(day, days))  # fetch this chunk
        day += dt.timedelta(days=days)  # move to the next chunk
        time.sleep(1)  # be polite to the API
    return pd.concat(chunks, ignore_index=True)  # all chunks in one table


def update_fires(end: dt.date | None = None) -> int:  # main step: returns how many new rows were added
    end = end or C.today_utc() - dt.timedelta(days=1)  # default: up to yesterday (today isn't complete yet)
    existing = pd.read_parquet(C.FIRES_RAW)  # all fires so far
    last = existing["acq_date"].max().date()  # newest day we already have
    start = min(last + dt.timedelta(days=1), end - dt.timedelta(days=C.FIRMS_DAYS - 1))  # catch up the gap, or re-check 5 days
    start = max(start, end - dt.timedelta(days=MAX_CATCHUP_DAYS - 1))  # but never more than 60 days back
    log.info("Fetching %s to %s (file ended %s)", start, end, last)  # record the window
    new = clean(fetch_window(start, end), existing)  # fetch and clean the whole window
    keys = ["latitude", "longitude", "acq_date", "acq_time"]  # one detection = same place, day and time
    combined = pd.concat([existing, new], ignore_index=True)  # old + new
    combined = combined.drop_duplicates(subset=keys, keep="first")  # drop re-fetched detections we already have
    added = len(combined) - len(existing)  # truly new rows
    combined = combined.sort_values(["acq_date", "acq_time"]).reset_index(drop=True)  # keep the file in time order
    tmp = C.FIRES_RAW.with_suffix(".tmp.parquet")  # temp file next to the real one
    combined.to_parquet(tmp, index=False)  # write the temp file fully first
    tmp.replace(C.FIRES_RAW)  # then swap it in (atomic on the same disk)
    log.info("Added %d new detections; total %d; last date %s", added, len(combined), combined["acq_date"].max().date())  # summary
    return added  # used by the runner's report


if __name__ == "__main__":  # run alone: python -m src.pipeline.fetch_fires
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(name)s | %(message)s")  # show log lines
    update_fires()  # run the update up to yesterday