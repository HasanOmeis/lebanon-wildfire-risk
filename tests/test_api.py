"""API tests: run with  python -m pytest tests/test_api.py -v"""  # file description and how to run it

import math  # checks for NaN values
from pathlib import Path  # file paths

import pytest  # test runner tools (fixtures, skip)
from fastapi.testclient import TestClient  # calls the app in Python, no server needed

from src.api.main import app  # the FastAPI app (nothing loads until the lifespan runs)

ROOT = Path(__file__).resolve().parents[1]  # project root (tests/test_api.py -> one folder up)
NEEDED = [  # files the API needs; they are rebuilt by scripts, not stored in Git
    ROOT / "models" / "lgbm_tuned.txt",  # the trees (src/train_final.py)
    ROOT / "models" / "risk_levels.json",  # the level cutoffs (src/risk_levels.py)
    ROOT / "data" / "serving" / "recent_features.parquet",  # serving features (src/export_serving.py)
    ROOT / "data" / "serving" / "grid.geojson",  # serving grid (src/export_serving.py)
]  # end of the list
pytestmark = pytest.mark.skipif(  # applies to every test in this file
    not all(p.exists() for p in NEEDED),  # skip if any needed file is missing...
    reason="model or serving files missing: run train_final.py, risk_levels.py and export_serving.py first",  # ...and say why
)  # end of the skip rule

LEVELS = ["low", "moderate", "high", "extreme"]  # valid level names, lowest first


@pytest.fixture(scope="module")  # build once and share with every test in this file (loading the model is slow)
def client():  # the test client
    with TestClient(app) as c:  # "with" runs the lifespan: model and store load like in production
        yield c  # hand the client to the tests; the lifespan closes when they finish


@pytest.fixture(scope="module")  # build once
def store(client):  # the serving data the app loaded (needs the client so the lifespan has run)
    return app.state.store  # the same FeatureStore object the endpoints use


@pytest.fixture(scope="module")  # build once
def feature_cols(client):  # the 52 feature names in model order
    return app.state.predictor.feature_cols  # read from the loaded predictor


def to_body(row, feature_cols) -> dict:  # one feature row -> the JSON body /predict expects
    features = {c: (None if math.isnan(float(row[c])) else float(row[c])) for c in feature_cols}  # NaN isn't valid JSON, send None
    return {"cell_id": int(row["cell_id"]), "date": str(row["date"]), "features": features}  # the full body


# ---------- system ----------

def test_health(client, feature_cols):  # /health answers and describes the model
    r = client.get("/health")  # call the endpoint
    assert r.status_code == 200  # success
    body = r.json()  # the JSON answer as a dict
    assert body["status"] == "ok"  # model loaded
    assert body["n_features"] == len(feature_cols) == 52  # the right number of features
    assert list(body["risk_cutoffs"]) == LEVELS[1:]  # moderate, high, extreme, in that order


def test_root_redirects_to_docs(client):  # the bare address sends people to /docs
    r = client.get("/", follow_redirects=False)  # don't follow, so we can see the redirect itself
    assert r.status_code in (302, 307)  # a redirect code
    assert r.headers["location"] == "/docs"  # pointing at the docs


# ---------- predict ----------

def test_predict_real_row(client, store, feature_cols):  # a real row gives a valid prediction with reasons
    row = store.rows_for(store.last).iloc[0]  # first cell on the latest day
    r = client.post("/predict", params={"reasons": "true"}, json=to_body(row, feature_cols))  # call with reasons
    assert r.status_code == 200  # success
    body = r.json()  # the answer
    assert 0 <= body["risk"] <= 1  # a probability
    assert body["level"] in LEVELS  # a valid level
    assert len(body["reasons"]) == 3  # top 3 reasons by default
    assert body["forecast_for"] > body["date"]  # forecast is for the next day (ISO dates compare correctly as text)


def test_predict_rejects_bad_features(client, store, feature_cols):  # a typo'd feature name gets a clear 422
    body = to_body(store.rows_for(store.last).iloc[0], feature_cols)  # a valid body...
    body["features"]["vpd_mx"] = body["features"].pop("vpd_max")  # ...with one name misspelled
    r = client.post("/predict", json=body)  # call it
    assert r.status_code == 422  # rejected
    assert "vpd_mx" in r.json()["detail"]  # the message names the unknown feature
    assert "vpd_max" in r.json()["detail"]  # and the missing one


def test_predict_rejects_negative_cell(client, store, feature_cols):  # Pydantic's ge=0 rule works
    body = to_body(store.rows_for(store.last).iloc[0], feature_cols)  # a valid body...
    body["cell_id"] = -1  # ...with an impossible cell
    assert client.post("/predict", json=body).status_code == 422  # rejected before our code runs


def test_batch_keeps_order(client, store, feature_cols):  # batch returns one answer per row, in the same order
    rows = store.rows_for(store.last).head(20)  # 20 cells
    r = client.post("/predict/batch", json={"rows": [to_body(x, feature_cols) for _, x in rows.iterrows()]})  # one call
    assert r.status_code == 200  # success
    body = r.json()  # the answer
    assert body["count"] == 20  # one prediction per row
    assert [p["cell_id"] for p in body["predictions"]] == rows["cell_id"].tolist()  # same order as sent


def test_batch_rejects_empty(client):  # min_length=1 on the rows list
    assert client.post("/predict/batch", json={"rows": []}).status_code == 422  # an empty batch is rejected


# ---------- map ----------

def test_risk_latest_map(client, store):  # the latest map covers every cell with valid values
    r = client.get("/risk/latest")  # call the endpoint
    assert r.status_code == 200  # success
    body = r.json()  # the answer
    assert body["date"] == str(store.last)  # it's the newest served day
    assert body["n_cells"] == 488  # every cell
    assert sum(body["level_counts"].values()) == 488  # the counts add up
    feats = body["geojson"]["features"]  # the map features
    assert len(feats) == 488  # one per cell
    assert {f["geometry"]["type"] for f in feats} <= {"Polygon", "MultiPolygon"}  # real shapes
    assert all(0 <= f["properties"]["risk"] <= 1 for f in feats)  # every risk is a probability
    lat, lon = feats[0]["properties"]["lat"], feats[0]["properties"]["lon"]  # one cell center
    assert 33.0 < lat < 34.8 and 35.0 < lon < 36.7  # inside Lebanon, in lat/lon (not UTM meters)


def test_risk_day_errors(client):  # bad days get the right error codes
    assert client.get("/risk/2010-01-01").status_code == 404  # a real date outside the window
    assert client.get("/risk/yesterday").status_code == 422  # not a date at all


# ---------- explain ----------

def test_explain(client):  # /explain returns the asked number of reasons
    r = client.get("/explain/226", params={"k": 5})  # 5 reasons for cell 226
    assert r.status_code == 200  # success
    reasons = r.json()["reasons"]  # the reasons
    assert len(reasons) == 5  # exactly k
    impacts = [abs(x["impact"]) for x in reasons]  # their strengths
    assert impacts == sorted(impacts, reverse=True)  # strongest first


def test_explain_errors(client):  # bad inputs get the right error codes
    assert client.get("/explain/9999").status_code == 404  # cell doesn't exist
    assert client.get("/explain/-1").status_code == 422  # negative cell
    assert client.get("/explain/226", params={"k": 11}).status_code == 422  # k above 10
    assert client.get("/explain/226", params={"day": "2010-01-01"}).status_code == 404  # day outside the window


# ---------- consistency ----------

def test_same_risk_everywhere(client, store, feature_cols):  # all four routes give the same risk for the same cell-day
    cell = 226  # the most fire-prone cell
    row = store.cell_row(store.last, cell).iloc[0]  # its features on the latest day
    via_predict = client.post("/predict", json=to_body(row, feature_cols)).json()["risk"]  # route 1
    via_batch = client.post("/predict/batch", json={"rows": [to_body(row, feature_cols)]}).json()["predictions"][0]["risk"]  # route 2
    feats = client.get("/risk/latest").json()["geojson"]["features"]  # route 3: the map
    via_map = next(f["properties"]["risk"] for f in feats if f["properties"]["cell_id"] == cell)  # this cell on the map
    via_explain = client.get(f"/explain/{cell}").json()["risk"]  # route 4
    assert via_predict == via_batch == via_map == via_explain  # identical: no route drifts from the others


def test_levels_rise_with_risk(client):  # the level function is monotonic: more risk never means a lower level
    p = app.state.predictor  # the loaded predictor
    cuts = list(p.cutoffs.values())  # moderate, high, extreme cutoffs
    probes = [0.0, cuts[0] - 1e-9, cuts[0], cuts[1], cuts[2], 1.0]  # just below/at each boundary, plus the extremes
    names = [p.level(x) for x in probes]  # level of each probe
    assert names == ["low", "low", "moderate", "high", "extreme", "extreme"]  # boundaries land on the higher level