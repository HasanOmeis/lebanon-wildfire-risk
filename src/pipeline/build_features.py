"""Pipeline step: re-join fires to cells, then rebuild the feature table up to a given day (same code as training)."""  # file description

import datetime as dt  # dates
import logging  # log lines for the runner
import subprocess  # runs the Day 3 and Day 7 scripts as child processes
import sys  # sys.executable = the venv's own python.exe

import pandas as pd  # reads the finished table to check it

from . import config as C  # pipeline settings (Step 2)

log = logging.getLogger("pipeline.features")  # this step's own logger


def run_script(*args: str) -> None:  # run one project script from the root; stop the pipeline if it fails
    cmd = [sys.executable, *args]  # same Python as the pipeline (the .venv one)
    log.info("Running: %s", " ".join(args))  # record which script runs
    result = subprocess.run(cmd, cwd=C.ROOT, capture_output=True, text=True)  # run it from the project root and capture its output
    for line in result.stdout.strip().splitlines()[-6:]:  # the script's last 6 lines are its summary
        log.info("  | %s", line)  # copy them into our log
    if result.returncode != 0:  # the script crashed
        log.error("Script failed:\n%s", result.stderr[-2000:])  # the end of its error message
        raise RuntimeError(f"{args[0]} failed with exit code {result.returncode}")  # stop the whole run


def build_features(day: dt.date) -> None:  # main step: features for every cell up to 'day'
    run_script("src/join_fires.py")  # Day 3: new fires -> cell_id + hotspot flag
    run_script("src/features.py", "--end", day.isoformat())  # Day 7: the exact training feature code, ending on 'day'
    last = pd.read_parquet(C.FEATURES, columns=["date"])["date"].max().date()  # newest day in the rebuilt table
    if last != day:  # the build didn't reach the requested day
        raise RuntimeError(f"features.parquet ends {last}, expected {day}")  # stop before scoring stale data
    log.info("Features ready up to %s", last)  # record success


if __name__ == "__main__":  # run alone: python -m src.pipeline.build_features
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(name)s | %(message)s")  # show log lines
    build_features(C.today_utc() - dt.timedelta(days=1))  # yesterday = last complete day