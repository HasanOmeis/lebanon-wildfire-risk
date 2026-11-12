"""Score the checked serving day with the API's own Predictor, save history, refresh the serving files, and publish them."""  # file description

import datetime as dt  # dates and the run timestamp
import json  # writes latest.json
import logging  # log lines for the runner

import pandas as pd  # tables

from src.api.predictor import Predictor  # Day 13: the exact model wrapper the API uses
from src.publish_hub import publish  # Day 17: uploads the serving files to Hugging Face

from . import config as C  # pipeline settings (Step 2)
from .build_features import run_script  # Step 6: runs a project script and stops on failure
from .notify_api import reload_api  # tells a running API (laptop or live) to reload

log = logging.getLogger("pipeline.score")  # this step's own logger


def score_day(rows: pd.DataFrame, day: dt.date) -> pd.DataFrame:  # risk for day + 1, from the features of 'day'
    p = Predictor()  # load model, metadata and level cutoffs (same as API startup)
    results = p.predict_frame(rows)  # [{"risk", "level", "reasons"}, ...] in row order
    out = pd.DataFrame({  # one tidy row per cell
        "cell_id": rows["cell_id"].to_numpy(),  # which cell
        "feature_date": pd.Timestamp(day),  # the complete day the features describe
        "forecast_date": pd.Timestamp(day + dt.timedelta(days=1)),  # the day the risk is for
        "risk": [r["risk"] for r in results],  # fire probability
        "level": [r["level"] for r in results],  # low / moderate / high / extreme
    })  # end of the table
    out["model_file"] = p.model_file  # which model made it (useful after retraining)
    out["trained_to"] = p.trained_to  # that model's training cutoff
    out["run_at"] = pd.Timestamp.now(tz="UTC")  # when this run happened
    return out.sort_values("risk", ascending=False).reset_index(drop=True)  # riskiest cells first


def save_history(pred: pd.DataFrame) -> None:  # one parquet per forecast day + a small summary file
    C.PREDICTIONS.mkdir(parents=True, exist_ok=True)  # data/predictions (Step 1), in case it was cleaned
    day = pred["forecast_date"].iloc[0].date()  # the forecast day
    path = C.PREDICTIONS / f"risk_{day}.parquet"  # e.g. risk_2026-11-10.parquet
    tmp = path.with_suffix(".tmp.parquet")  # temp file next to it
    pred.to_parquet(tmp, index=False)  # write fully first
    tmp.replace(path)  # then swap in; rerunning the same day simply replaces it (idempotent)

    summary = {  # small JSON for the map page and quick checks
        "forecast_date": str(day),  # the day the risk is for
        "feature_date": str(pred["feature_date"].iloc[0].date()),  # the day the features describe
        "model_file": pred["model_file"].iloc[0],  # which model
        "levels": pred["level"].value_counts().to_dict(),  # cells per level
        "max_risk": round(float(pred["risk"].max()), 4),  # highest risk in the country
        "top_cells": pred.head(5)[["cell_id", "risk", "level"]].to_dict("records"),  # the 5 riskiest cells
        "run_at": pred["run_at"].iloc[0].isoformat(),  # when it ran
    }  # end of the summary
    (C.PREDICTIONS / "latest.json").write_text(json.dumps(summary, indent=2))  # overwrite with today's summary
    log.info("Saved %s and latest.json; levels %s", path.name, summary["levels"])  # record it


def publish_serving() -> None:  # refresh data/serving, reload the laptop API, upload to Hugging Face, reload the live API
    run_script("src/export_serving.py")  # Day 13 script, unchanged: grid.geojson + last 400 days of features
    last = pd.read_parquet(C.SERVING / "recent_features.parquet", columns=["date"])["date"].max().date()  # newest served day
    log.info("Serving files refreshed; newest served day %s", last)  # the API's /risk/latest will show last + 1
    reload_api("API_URL")  # 1. the laptop API swaps in the new files now (a warning only if it's down)

    try:  # 2. upload to Hugging Face (Day 17); a failure must not undo the local update
        publish(f"Serving update: features to {last}")  # one commit with all 5 files, named after the newest day
    except Exception as e:  # no internet, Hub down, token expired or missing...
        log.warning("Hugging Face upload failed (%s: %s); the live map keeps yesterday's data", type(e).__name__, e)  # warn, don't fail the run
        return  # nothing new on the Hub, so there's no point waking the live API
    reload_api("LIVE_API_URL")  # 3. the live API (Render) pulls the new upload; skipped until LIVE_API_URL is set in .env


def score_and_publish(rows: pd.DataFrame, day: dt.date) -> pd.DataFrame:  # main step: score, save, publish
    pred = score_day(rows, day)  # 1. score
    save_history(pred)  # 2-3. history file + summary
    publish_serving()  # 4. refresh files, reload laptop API, upload, reload live API (last, after everything else succeeded)
    return pred  # for the runner's report


if __name__ == "__main__":  # run alone: python -m src.pipeline.score
    from .checks import check_day  # Step 7 quality gate
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(name)s | %(message)s")  # show log lines
    day = C.today_utc() - dt.timedelta(days=1)  # yesterday = last complete day
    pred = score_and_publish(check_day(day), day)  # checks first, then score and publish
    print(pred.head(5).to_string(index=False))  # the 5 riskiest cells for today