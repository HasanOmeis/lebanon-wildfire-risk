"""Pipeline tests: quality gate, idempotent fetches, weather upsert rules, catch-up days and /reload security (offline)."""  # file description

import datetime as dt  # dates

import numpy as np  # NaN values
import pandas as pd  # small fake tables
import pytest  # test runner: fixtures, raises, monkeypatch
from fastapi.testclient import TestClient  # calls the API in-process, no server needed

from src.api.main import app  # the FastAPI app (for /reload)
from src.pipeline import checks, config, fetch_fires, fetch_weather  # the modules under test
from src.pipeline.run_daily import days_to_score  # catch-up day list

DAY = dt.date(2026, 11, 9)  # a fixed test day, so results never depend on today's date
FEATS = ["tmax", "rh_min", "precip", "wind_max", "gust_max", "et0", "vpd_max",  # raw weather (core)
         "fire_today", "fires_last_7d", "days_since_fire", "nbr_fires_today", "ndvi_age_days"]  # fire history (core) + NDVI age
WX_VARS = fetch_weather.DAILY_VARS  # the 7 Open-Meteo column names


# ---------- 1. quality gate ----------

def make_day(day=DAY, n=488) -> pd.DataFrame:  # a valid serving day: n cells, sensible values
    df = pd.DataFrame({"cell_id": range(n), "date": pd.Timestamp(day)})  # one row per cell
    for col in FEATS:  # every feature...
        df[col] = 1.0  # ...a harmless value
    df["tmax"], df["rh_min"], df["ndvi_age_days"] = 20.0, 50.0, 10.0  # realistic weather and NDVI age
    return df  # the fake day


@pytest.fixture  # gives each test a writer that saves a fake feature table in a temp folder
def write_features(tmp_path, monkeypatch):  # tmp_path = empty folder, monkeypatch = safe temporary replacements
    path = tmp_path / "features.parquet"  # fake features file
    monkeypatch.setattr(config, "FEATURES", path)  # checks.py now reads the fake file
    monkeypatch.setattr(checks, "model_features", lambda: FEATS)  # and expects our short feature list
    return lambda df: df.to_parquet(path, index=False)  # the test calls this to write its table


def test_good_day_passes(write_features):  # a complete, clean day
    write_features(make_day())  # save it
    rows = checks.check_day(DAY)  # run the gate
    assert len(rows) == 488  # every cell came back


@pytest.mark.parametrize("break_it, message", [  # each case breaks the day in one way
    (lambda df: df.iloc[:-1], "cells"),  # one cell missing
    (lambda df: df.assign(tmax=np.where(df.index == 0, np.nan, df["tmax"])), "missing core"),  # one core value missing
    (lambda df: df.assign(rh_min=np.where(df.index == 0, 150.0, df["rh_min"])), "out of range"),  # humidity above 100%
    (lambda df: df.drop(columns="vpd_max"), "missing from the table"),  # a model feature column gone
])  # end of the cases
def test_bad_day_is_blocked(write_features, break_it, message):  # every broken day must be stopped
    write_features(break_it(make_day()))  # save the broken day
    with pytest.raises(checks.CheckError, match=message):  # the gate must raise, with the right reason
        checks.check_day(DAY)  # run the gate


def test_stale_day_is_blocked(write_features):  # asking for a day the table doesn't have
    write_features(make_day())  # table ends on DAY
    with pytest.raises(checks.CheckError, match="No feature rows"):  # the day after must be refused
        checks.check_day(DAY + dt.timedelta(days=1))  # one day too far


# ---------- 2. fires: re-fetching never duplicates ----------

def test_fire_update_is_idempotent(tmp_path, monkeypatch):  # overlapping downloads add each detection only once
    existing = pd.DataFrame({  # one detection already in the file
        "latitude": [33.5], "longitude": [35.5], "acq_date": pd.to_datetime(["2026-11-01"]),  # where and when
        "acq_time": [1100], "daynight": ["D"], "frp": [3.0], "bright_ti4": [330.0],  # pass time and fire values
        "confidence": ["n"], "source": ["VIIRS_SNPP_NRT"],  # cleaned confidence and source label
    })  # end of the existing table
    path = tmp_path / "firms.parquet"  # fake fire file
    existing.to_parquet(path, index=False)  # save it
    monkeypatch.setattr(config, "FIRES_RAW", path)  # fetch_fires now uses the fake file

    download = pd.DataFrame({  # what the fake FIRMS returns, in the API's raw format
        "latitude": [33.5, 34.1, 33.9], "longitude": [35.5, 35.9, 35.7],  # 1st = the existing one again
        "acq_date": ["2026-11-01", "2026-11-03", "2026-11-03"], "acq_time": [1100, 1030, 1030],  # 2nd = new, 3rd = low confidence
        "daynight": ["D", "D", "D"], "frp": [3.0, 5.0, 1.0], "bright_ti4": [330.0, 340.0, 300.0],  # fire values
        "confidence": ["nominal", "high", "low"], "satellite": ["N", "N", "N"],  # long names + an extra API column
    })  # end of the fake download
    monkeypatch.setattr(fetch_fires, "fetch_window", lambda start, end: download.copy())  # no internet: return the fake table

    assert fetch_fires.update_fires(end=DAY) == 1  # first run: only the new, confident detection is added
    assert fetch_fires.update_fires(end=DAY) == 0  # second run: nothing new
    saved = pd.read_parquet(path)  # the file after both runs
    assert len(saved) == 2 and saved["confidence"].isin(["n", "h"]).all()  # 2 rows, low confidence dropped, short codes


# ---------- 3. weather: forecasts replaced, archive protected ----------

def test_weather_upsert_rules(tmp_path, monkeypatch):  # forecasts refresh, archive rows never change
    def wx_rows(cells, dates, value, source):  # helper: weather rows with one value everywhere
        df = pd.DataFrame([(c, pd.Timestamp(d)) for c in cells for d in dates], columns=["cell_id", "date"])  # cell x date
        for v in WX_VARS:  # every weather column
            df[v] = value  # same value
        df["relative_humidity_2m_min"] = df["relative_humidity_2m_min"].astype("int64")  # whole numbers, like the real file
        df["source"] = source  # archive or forecast
        return df  # the rows

    existing = pd.concat([  # what the file holds before the update
        wx_rows([0, 1], ["2026-11-01"], 10.0, "archive"),  # final archive day
        wx_rows([0, 1], ["2026-11-02"], 15.0, "forecast"),  # an older forecast
    ], ignore_index=True)  # one table
    wx_path = tmp_path / "weather.parquet"  # fake weather file
    existing.to_parquet(wx_path, index=False)  # save it
    pd.DataFrame({"wp_lat": [33.1], "wp_lon": [35.1], "wp_id": [0]}).to_csv(tmp_path / "points.csv", index=False)  # one weather point
    pd.DataFrame({"cell_id": [0, 1], "wp_id": [0, 0]}).to_csv(tmp_path / "links.csv", index=False)  # both cells use it
    for name, value in [("WEATHER", wx_path), ("WEATHER_POINTS", tmp_path / "points.csv"),  # point config at the fakes...
                        ("CELL_TO_POINT", tmp_path / "links.csv"), ("EXPECTED_CELLS", 2), ("WEATHER_PAST_DAYS", 3)]:  # ...and shrink the grid
        monkeypatch.setattr(config, name, value)  # temporary change, undone after the test
    monkeypatch.setattr(fetch_weather.time, "sleep", lambda s: None)  # no waiting in tests

    def fake_api(value):  # builds a fake Open-Meteo that returns 'value' for every day asked
        def fetch(points, start, end):  # same signature as the real fetch_points
            days = pd.date_range(start, end).strftime("%Y-%m-%d")  # the requested days as text, like the API
            df = pd.DataFrame({"time": days, "wp_id": points["wp_id"].iloc[0]})  # one row per day
            for v in WX_VARS:  # every weather column
                df[v] = value  # the same value
            return df  # the fake reply
        return fetch  # the fake function

    end = dt.date(2026, 11, 3)  # update up to Nov 3 (window Nov 1-3)
    monkeypatch.setattr(fetch_weather, "fetch_points", fake_api(25.0))  # first update returns 25
    fetch_weather.update_weather(end=end)  # run it
    monkeypatch.setattr(fetch_weather, "fetch_points", fake_api(30.0))  # second update returns fresher values: 30
    fetch_weather.update_weather(end=end)  # run it again

    out = pd.read_parquet(wx_path).set_index(["date", "cell_id"])  # the file after both runs
    temp = out["temperature_2m_max"]  # one column is enough to check
    assert len(out) == 6  # 2 cells x 3 days: nothing duplicated
    assert (temp.loc["2026-11-01"] == 10.0).all()  # archive day untouched
    assert (temp.loc["2026-11-02"] == 30.0).all() and (temp.loc["2026-11-03"] == 30.0).all()  # forecasts = newest values
    assert (out.loc["2026-11-03", "source"] == "forecast").all()  # new rows are tagged


# ---------- 4. catch-up days ----------

def test_days_to_score():  # which feature days a run scores
    assert days_to_score(None, DAY) == [DAY]  # normal run: just the last day
    assert days_to_score(dt.date(2026, 11, 7), DAY) == [dt.date(2026, 11, 7), dt.date(2026, 11, 8), DAY]  # catch-up, oldest first
    with pytest.raises(ValueError):  # a start after the last complete day...
        days_to_score(DAY + dt.timedelta(days=1), DAY)  # ...must be refused


# ---------- 5. /reload security ----------

def test_reload_security(monkeypatch):  # 503 without a token, 401 with a wrong one, 200 with the right one
    monkeypatch.delenv("RELOAD_TOKEN", raising=False)  # start with no token set
    with TestClient(app) as client:  # starts the app (loads the real model and serving files)
        assert client.post("/reload").status_code == 503  # switched off
        monkeypatch.setenv("RELOAD_TOKEN", "test-secret")  # now the server "has" a token
        assert client.post("/reload", headers={"X-Reload-Token": "wrong"}).status_code == 401  # wrong token refused
        ok = client.post("/reload", headers={"X-Reload-Token": "test-secret"})  # right token
        assert ok.status_code == 200 and ok.json()["status"] == "reloaded"  # reloaded