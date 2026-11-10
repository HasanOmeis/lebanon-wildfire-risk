"""Daily pipeline runner: fetch -> features -> checks -> score -> publish, with logs, timing, lock and run history."""  # file description

import argparse  # command-line options (--from, --skip-fetch)
import csv  # appends a line to history.csv
import datetime as dt  # dates and timestamps
import logging  # log lines to the terminal and a file
import os  # process id for the lock file
import sys  # exit codes
import time  # step timing

from . import config as C  # pipeline settings (Step 2)
from .build_features import build_features  # Step 6
from .checks import CheckError, check_day  # Step 7
from .fetch_fires import update_fires  # Step 3
from .fetch_ndvi import check_terrain, update_ndvi  # Step 5
from .fetch_weather import update_weather  # Step 4
from .score import score_and_publish, save_history, score_day  # Step 8

log = logging.getLogger("pipeline.run")  # the runner's own logger
LOCK = C.RUNS / ".lock"  # exists while a run is in progress
STALE_LOCK_HOURS = 3  # an older lock is left over from a crash
LOG_FORMAT = "%(asctime)s %(levelname)-7s %(name)s | %(message)s"  # same format as every step


def setup_logging(started: dt.datetime) -> str:  # log to the terminal and to this run's own file
    C.RUNS.mkdir(parents=True, exist_ok=True)  # pipeline_runs (Step 1)
    log_file = C.RUNS / f"run_{started:%Y-%m-%d_%H%M%S}.log"  # e.g. run_2026-11-10_203000.log (UTC time)
    logging.basicConfig(  # one setup for every logger in the pipeline
        level=logging.INFO,  # INFO and above
        format=LOG_FORMAT,  # time, level, step, message
        handlers=[logging.StreamHandler(), logging.FileHandler(log_file, encoding="utf-8")],  # terminal + file
    )  # end of the setup
    return log_file.name  # for the history line


def take_lock() -> None:  # stop if another run is already going
    if LOCK.exists():  # someone holds the lock
        age_h = (time.time() - LOCK.stat().st_mtime) / 3600  # how old it is, in hours
        if age_h < STALE_LOCK_HOURS:  # a real run is probably still going
            raise RuntimeError(f"Another run holds the lock ({LOCK.read_text().strip()}, {age_h:.1f} h old)")  # don't overlap
        log.warning("Removing stale lock (%.1f h old)", age_h)  # left over from a crash
    LOCK.write_text(f"pid {os.getpid()} since {dt.datetime.now(dt.timezone.utc):%H:%M} UTC")  # claim it


def timed(name: str, func, *args):  # run one step and log how long it took
    t0 = time.perf_counter()  # start the clock
    result = func(*args)  # run the step
    log.info("Step %-9s done in %5.1f s", name, time.perf_counter() - t0)  # e.g. "Step weather done in 11.2 s"
    return result  # pass the step's result back


def days_to_score(start: dt.date | None, last: dt.date) -> list[dt.date]:  # which feature days to score, oldest first
    first = start or last  # normal run: just the last day
    if first > last:  # a future or not-yet-complete day was asked for
        raise ValueError(f"--from {first} is after the last complete day {last}")  # clear message
    return [first + dt.timedelta(days=i) for i in range((last - first).days + 1)]  # every day in between


def write_history(row: dict) -> None:  # one line per run in history.csv
    path = C.RUNS / "history.csv"  # the run history file
    new = not path.exists()  # first run ever needs a header
    with path.open("a", newline="", encoding="utf-8") as f:  # append mode
        w = csv.DictWriter(f, fieldnames=list(row))  # columns from the dict keys
        if new:  # first time
            w.writeheader()  # write the column names
        w.writerow(row)  # this run's line


def main() -> int:  # returns the exit code
    parser = argparse.ArgumentParser(description="Run the daily wildfire-risk pipeline.")  # command-line options
    parser.add_argument("--from", dest="start", type=dt.date.fromisoformat, default=None,  # catch-up start day...
                        help="score every day from this feature day (YYYY-MM-DD) up to yesterday")  # ...and its help text
    parser.add_argument("--skip-fetch", action="store_true", help="reuse the data on disk (offline testing)")  # no downloads
    args = parser.parse_args()  # read what was typed

    started = dt.datetime.now(dt.timezone.utc)  # run start time (UTC)
    log_name = setup_logging(started)  # terminal + file logging
    last = C.today_utc() - dt.timedelta(days=1)  # last complete day
    status, error, scored = "ok", "", []  # filled in as the run goes
    try:  # everything that can fail
        take_lock()  # no overlapping runs
        days = days_to_score(args.start, last)  # which days to score
        log.info("Run started: features up to %s, scoring %d day(s)", last, len(days))  # what this run will do
        if not args.skip_fetch:  # normal mode: download fresh data
            timed("fires", update_fires, last)  # Step 3: FIRMS up to yesterday
            timed("weather", update_weather, C.today_utc())  # Step 4: weather up to today
            timed("ndvi", update_ndvi, C.today_utc())  # Step 5: new composites up to today
        timed("terrain", check_terrain)  # Step 5: static data still intact
        timed("features", build_features, last)  # Step 6: rebuild up to yesterday
        for day in days[:-1]:  # catch-up days (all but the newest): score and save, no publish yet
            pred = timed("score", score_day, timed("checks", check_day, day), day)  # Steps 7-8 for an older day
            save_history(pred)  # its history file
            scored.append(str(day))  # remember it
        timed("publish", score_and_publish, timed("checks", check_day, last), last)  # newest day: score, save, publish
        scored.append(str(last))  # remember it
    except CheckError as e:  # blocked by the quality gate: nothing new was published
        status, error = "blocked", str(e)  # record why
        log.error("BLOCKED by data check: %s (yesterday's map stays up)", e)  # loud message
    except Exception as e:  # any other failure
        status, error = "failed", f"{type(e).__name__}: {e}"  # record why
        log.exception("Run FAILED")  # message + full traceback in the log
    finally:  # always runs, success or not
        if LOCK.exists() and status != "locked":  # release our lock
            LOCK.unlink()  # delete the lock file
        secs = (dt.datetime.now(dt.timezone.utc) - started).total_seconds()  # total run time
        write_history({  # one line in history.csv
            "started_utc": f"{started:%Y-%m-%d %H:%M:%S}",  # when it began
            "status": status,  # ok / blocked / failed
            "seconds": round(secs, 1),  # how long it took
            "scored_days": " ".join(scored),  # which feature days were scored
            "log_file": log_name,  # where the details are
            "error": error[:300],  # short reason if it didn't succeed
        })  # end of the history line
        log.info("Run %s in %.1f s", status.upper(), secs)  # final line
    return {"ok": 0, "blocked": 2}.get(status, 1)  # exit code for Task Scheduler


if __name__ == "__main__":  # run: python -m src.pipeline.run_daily
    sys.exit(main())  # pass the exit code to Windows