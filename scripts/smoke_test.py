"""Smoke test: checks every endpoint of a running API (local container or deployed URL)."""  # file description

import math  # isnan, to turn missing values into None
import sys  # command-line argument and exit code
from pathlib import Path  # file paths

import httpx  # HTTP client that calls the API
import lightgbm as lgb  # reads the feature names from the model file
import pandas as pd  # reads the serving features

ROOT = Path(__file__).resolve().parents[1]  # project root (scripts/smoke_test.py -> one folder up)
BASE_URL = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000"  # API address: first argument, or the local container
TEST_CELL = 464  # a cell we know exists (it was "extreme" on 2026-10-30)

results = []  # list of (check name, passed?) collected as we go


def check(name: str, passed: bool, detail: str = "") -> None:  # records and prints one check
    results.append((name, passed))  # remember it for the summary
    print(f"{'PASS' if passed else 'FAIL'}  {name}  {detail}")  # one line per check


def clean(value) -> float | None:  # turns one parquet value into something JSON can carry
    if value is None or (isinstance(value, float) and math.isnan(value)):  # missing value
        return None  # sent as null; the model treats it as missing
    return float(value)  # numpy numbers and categories become plain floats


feature_names = lgb.Booster(model_file=str(ROOT / "models" / "lgbm_tuned.txt")).feature_name()  # the 52 names, in the model's own order
df = pd.read_parquet(ROOT / "data" / "serving" / "recent_features.parquet")  # same serving file that's inside the image
df["date"] = pd.to_datetime(df["date"]).dt.date  # plain dates, like the API uses
last_day = max(df["date"])  # newest day in the window
day_rows = df[df["date"] == last_day]  # all 488 cells on that day


def row_body(row) -> dict:  # builds one CellFeatures request body from a table row
    return {  # the shape defined in schemas.py
        "cell_id": int(row["cell_id"]),  # which cell
        "date": str(last_day),  # day t as text, e.g. "2026-10-29"
        "features": {name: clean(row[name]) for name in feature_names},  # every model feature by name
    }  # end of the body


with httpx.Client(base_url=BASE_URL, timeout=60) as api:  # one connection reused for all calls; 60 s for slow networks
    print(f"Smoke test against {BASE_URL}\n")  # show which server we're testing

    r = api.get("/health")  # 1) is the service alive?
    h = r.json()  # the health answer as a dict
    check("health ok", r.status_code == 200 and h.get("status") == "ok", f"model={h.get('model_file')}")  # alive and model loaded
    check("same features as local model", h.get("n_features") == len(feature_names), f"{h.get('n_features')} vs {len(feature_names)}")  # container serves the same model

    r = api.get("/risk/latest")  # 2) the full map
    m = r.json()  # the map answer as a dict
    check("risk map has 488 cells", r.status_code == 200 and m["n_cells"] == 488 == len(m["geojson"]["features"]), f"{m['n_cells']} cells")  # summary and GeoJSON agree
    check("level counts add up", sum(m["level_counts"].values()) == m["n_cells"], str(m["level_counts"]))  # every cell has exactly one level
    check("timing header present", "x-process-time-ms" in r.headers, f"{r.headers.get('x-process-time-ms')} ms")  # middleware from Day 13 Step 13

    r = api.get(f"/explain/{TEST_CELL}")  # 3) one cell with its reasons
    e = r.json()  # the explain answer as a dict
    check("explain returns 3 reasons", r.status_code == 200 and len(e["reasons"]) == 3, f"risk={e['risk']} level={e['level']}")  # default k=3

    test_row = day_rows[day_rows["cell_id"] == TEST_CELL].iloc[0]  # 4) the test cell's raw features from the parquet
    r = api.post("/predict", json=row_body(test_row))  # score it from raw features
    p = r.json()  # the prediction as a dict
    check("predict matches explain", r.status_code == 200 and abs(p["risk"] - e["risk"]) < 1e-6, f"{p['risk']} vs {e['risk']}")  # same input, same answer

    batch = {"rows": [row_body(row) for _, row in day_rows.iterrows()]}  # 5) all 488 cells in one request
    r = api.post("/predict/batch", json=batch)  # score them together
    check("batch scores 488 cells", r.status_code == 200 and r.json()["count"] == 488, f"status {r.status_code}")  # one prediction per row

    r = api.get("/explain/9999")  # 6a) a cell that doesn't exist
    check("unknown cell -> 404", r.status_code == 404, r.json().get("detail", ""))  # clean "not found"
    r = api.get("/risk/1999-01-01")  # 6b) a date outside the window
    check("unknown date -> 404", r.status_code == 404, r.json().get("detail", ""))  # clean "not found" with the valid range
    bad = row_body(test_row)  # 6c) start from a valid body...
    bad["features"]["vpd_mx"] = bad["features"].pop("vpd_max")  # ...and misspell one feature name
    r = api.post("/predict", json=bad)  # send the broken request
    check("misspelled feature -> 422", r.status_code == 422, str(r.json().get("detail", ""))[:80])  # rejected, not silently wrong

passed = sum(ok for _, ok in results)  # number of passing checks
print(f"\n{passed}/{len(results)} checks passed")  # summary line
sys.exit(0 if passed == len(results) else 1)  # exit code 1 on any failure, so scripts and CI can stop