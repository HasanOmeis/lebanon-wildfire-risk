"""Sends real feature rows to the running API: python src/call_api.py predict | batch [YYYY-MM-DD]"""  # file description and usage

import json  # reads model_meta.json and prints answers
import sys  # reads the words typed after the script name
from pathlib import Path  # file paths

import httpx  # HTTP client (installed in Step 1)
import pandas as pd  # reads the feature table

ROOT = Path(__file__).resolve().parents[1]  # project root (src/call_api.py -> one folder up)
API = "http://127.0.0.1:8000"  # where uvicorn is running
FEATURES_PATH = ROOT / "data" / "processed" / "features.parquet"  # Day 7 feature table
FEATURE_COLS = json.loads((ROOT / "models" / "model_meta.json").read_text())["feature_cols"]  # the 52 names the model expects


def rows_for_day(day: str | None) -> pd.DataFrame:  # all 488 cells on one day (latest day if none given)
    df = pd.read_parquet(FEATURES_PATH)  # the full table
    dates = pd.to_datetime(df["date"])  # dates as real datetimes
    target = pd.Timestamp(day) if day else dates.max()  # the asked day, or the latest
    return df[dates == target].reset_index(drop=True)  # only that day's rows


def to_request(row: pd.Series) -> dict:  # one table row -> the JSON body /predict expects
    values = row[FEATURE_COLS].astype(float)  # the 52 features as plain numbers
    features = {k: (None if pd.isna(v) else float(v)) for k, v in values.items()}  # NaN isn't valid JSON, so send None
    return {"cell_id": int(row["cell_id"]), "date": f"{pd.Timestamp(row['date']):%Y-%m-%d}", "features": features}  # the full body


if __name__ == "__main__":  # runs only when you call this file directly
    mode = sys.argv[1] if len(sys.argv) > 1 else "predict"  # first word after the script name: predict or batch
    day = sys.argv[2] if len(sys.argv) > 2 else None  # optional second word: a date
    rows = rows_for_day(day)  # that day's 488 cells
    if mode == "predict":  # one cell, with reasons
        row = rows.loc[rows["cell_fire_rate"].idxmax()]  # the cell that burns most often historically: an interesting example
        answer = httpx.post(f"{API}/predict", params={"reasons": "true"}, json=to_request(row), timeout=30)  # call the API
        print("Status:", answer.status_code)  # 200 means success
        print(json.dumps(answer.json(), indent=2))  # the prediction, readable
    elif mode == "batch":  # every cell in one call (Step 8)
        body = {"rows": [to_request(r) for _, r in rows.iterrows()]}  # 488 rows in one body
        answer = httpx.post(f"{API}/predict/batch", json=body, timeout=60)  # call the API
        print("Status:", answer.status_code)  # 200 means success
        preds = pd.DataFrame(answer.json()["predictions"])  # the answers as a table
        print(f"{len(preds)} predictions for {preds['date'].iloc[0]} -> {preds['forecast_for'].iloc[0]}")  # count and days
        print(preds["level"].value_counts())  # cells at each level
        print(preds.nlargest(5, "risk")[["cell_id", "risk", "level"]])  # the 5 riskiest cells
    else:  # anything else typed
        print("Usage: python src/call_api.py predict | batch [YYYY-MM-DD]")  # how to use the script