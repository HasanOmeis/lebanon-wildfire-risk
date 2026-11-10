"""Quality gate: hard checks on the serving day's feature rows before anything is scored or published."""  # file description

import datetime as dt  # dates
import json  # reads model_meta.json
import logging  # log lines for the runner

import pandas as pd  # tables

from . import config as C  # pipeline settings (Step 2)

log = logging.getLogger("pipeline.checks")  # this step's own logger

CORE = [  # features that must never be missing on the serving day
    "tmax", "rh_min", "precip", "wind_max", "gust_max", "et0", "vpd_max",  # today's raw weather (fetch problems show here)
    "fire_today", "fires_last_7d", "days_since_fire", "nbr_fires_today",  # fire history (fire/join problems show here)
]  # end of the list
RANGES = {  # physical limits: (lowest allowed, highest allowed)
    "tmax": (-30, 55),  # °C, Lebanon's records are well inside this
    "rh_min": (0, 100),  # % humidity
    "precip": (0, 500),  # mm of rain in a day
    "wind_max": (0, 250),  # km/h
    "gust_max": (0, 300),  # km/h
    "vpd_max": (0, 15),  # kPa
}  # end of the ranges
NDVI_AGE_WARN = 45  # days; older NDVI is normal between composites but worth a log line


class CheckError(RuntimeError):  # its own error type, so the runner can say "blocked by a data check"
    pass  # nothing extra needed


def model_features() -> list[str]:  # the exact feature list the deployed model was trained on
    return json.loads(C.MODEL_META.read_text())["feature_cols"]  # same file the API uses


def check_day(day: dt.date) -> pd.DataFrame:  # run every check on one day; return its rows if all pass
    df = pd.read_parquet(C.FEATURES, filters=[("date", "==", pd.Timestamp(day))])  # read only that day's rows from the big table
    feats = model_features()  # what the model needs

    if df.empty:  # 1. fresh: is the day there at all?
        raise CheckError(f"No feature rows for {day}; the build did not reach it")  # stop
    if len(df) != C.EXPECTED_CELLS or df["cell_id"].nunique() != C.EXPECTED_CELLS:  # 2. complete: one row per cell
        raise CheckError(f"{day}: {len(df)} rows / {df['cell_id'].nunique()} cells, expected {C.EXPECTED_CELLS}")  # stop
    missing_cols = [c for c in feats if c not in df.columns]  # 3. schema: model columns not in the table
    if missing_cols:  # any at all is fatal
        raise CheckError(f"Features missing from the table: {missing_cols}")  # stop
    core_nan = {c: int(df[c].isna().sum()) for c in CORE if df[c].isna().any()}  # 4. core features with NaN
    if core_nan:  # any core gap is fatal
        raise CheckError(f"{day}: missing core values {core_nan}")  # stop
    for col, (lo, hi) in RANGES.items():  # 5. physical ranges
        bad = df[(df[col] < lo) | (df[col] > hi)]  # rows outside the limits
        if not bad.empty:  # any at all is fatal
            raise CheckError(f"{day}: {col} out of range [{lo}, {hi}] in {len(bad)} cells, e.g. {bad[col].iloc[0]}")  # stop

    other_nan = df[feats].isna().sum()  # soft: NaN count per feature
    other_nan = other_nan[other_nan > 0]  # keep only features that have some
    if not other_nan.empty:  # warn, LightGBM handles NaN
        log.warning("%s: NaN in non-core features %s", day, other_nan.to_dict())  # which ones and how many
    age = df["ndvi_age_days"].median()  # soft: how old the vegetation data is
    if age > NDVI_AGE_WARN:  # older than usual
        log.warning("%s: NDVI is %d days old (median); new composites may be delayed", day, age)  # record it

    log.info("Checks passed for %s: %d cells, %d features, NDVI age %d days, max tmax %.1f °C",  # summary line...
             day, len(df), len(feats), age, df["tmax"].max())  # ...values
    return df.reset_index(drop=True)  # the validated rows, ready for scoring


if __name__ == "__main__":  # run alone: python -m src.pipeline.checks
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(name)s | %(message)s")  # show log lines
    check_day(C.today_utc() - dt.timedelta(days=1))  # yesterday should pass
    try:  # today must FAIL: features stop at yesterday on purpose
        check_day(C.today_utc())  # no rows for today
    except CheckError as e:  # the expected failure
        log.info("Expected failure for today: %s", e)  # proves the gate really blocks