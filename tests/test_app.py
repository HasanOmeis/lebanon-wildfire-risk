"""Tests for the Streamlit map: the API client (fake HTTP) and the page itself (Streamlit AppTest, fake API data)."""  # file description

import datetime as dt  # dates for fresh/stale fake data
from pathlib import Path  # absolute path to the app script
from unittest.mock import MagicMock, patch  # fake objects and temporary replacements

import pytest  # test framework
import requests  # its exception classes, to simulate network failures
import streamlit as st  # to clear Streamlit's cache between tests
from streamlit.testing.v1 import AppTest  # runs a Streamlit script headlessly inside pytest

from app import api_client  # the module under test (Step 2)
from app.api_client import ApiError  # the client's single error type

APP_FILE = str(Path(__file__).resolve().parents[1] / "app" / "streamlit_app.py")  # absolute path, so AppTest finds it from anywhere
TODAY = dt.datetime.now(dt.timezone.utc).date()  # today in UTC, same rule as the page's freshness check


# ---------- fake data shaped exactly like the real API answers ----------

def square(lon: float, lat: float) -> dict:  # a small square polygon around a point (stands in for a ~5 km cell)
    d = 0.02  # half the side length in degrees
    ring = [[lon - d, lat - d], [lon + d, lat - d], [lon + d, lat + d], [lon - d, lat + d], [lon - d, lat - d]]  # closed ring: first point = last point
    return {"type": "Polygon", "coordinates": [ring]}  # GeoJSON polygon


def fake_risk(forecast_for: dt.date) -> dict:  # a 3-cell /risk answer; forecast_for decides fresh vs stale
    cells = [(0, 0.0005, "low"), (1, 0.005, "moderate"), (2, 0.02, "high")]  # (cell_id, risk, level)
    features = [  # one GeoJSON feature per cell
        {"type": "Feature", "geometry": square(35.5 + 0.1 * cid, 33.5),  # cells side by side
         "properties": {"cell_id": cid, "lat": 33.5, "lon": 35.5 + 0.1 * cid, "land_frac": 1.0, "risk": risk, "level": level}}  # same keys as the API
        for cid, risk, level in cells  # build all three
    ]  # end of the features
    return {  # the full answer
        "date": (forecast_for - dt.timedelta(days=1)).isoformat(),  # features day = forecast day - 1
        "forecast_for": forecast_for.isoformat(),  # the day the risk is about
        "n_cells": 3,  # three cells
        "level_counts": {"low": 1, "moderate": 1, "high": 1, "extreme": 0},  # matches the cells above
        "geojson": {"type": "FeatureCollection", "features": features},  # the map data
    }  # end of the answer


FAKE_HEALTH = {  # a /health answer
    "status": "ok", "model_file": "lgbm_tuned.txt", "trained_to": "2026-10-28", "n_features": 52,  # model info
    "risk_cutoffs": {"moderate": 0.003181, "high": 0.010286, "extreme": 0.026611},  # the real Day 13 cutoffs
}  # end of the answer


def fake_explain(cell_id: int, day=None, k: int = 3) -> dict:  # an /explain answer for any cell (same signature as the client)
    return {  # the answer
        "cell_id": cell_id, "date": "2026-11-10", "forecast_for": "2026-11-11", "risk": 0.02, "level": "high",  # the cell's risk
        "reasons": [  # top 3 SHAP reasons
            {"feature": "days_since_fire", "label": "days since the last fire here", "value": 12.0, "impact": 0.8, "direction": "raises risk"},  # pushes up
            {"feature": "burn_freq", "label": "how often this area burns", "value": 0.0039, "impact": 0.5, "direction": "raises risk"},  # small value
            {"feature": "rain_30d", "label": "rain over the last month", "value": 40.0, "impact": -0.3, "direction": "lowers risk"},  # pushes down
        ],  # end of the reasons
    }  # end of the answer


def fake_response(status: int, body=None, text: str = "") -> MagicMock:  # a stand-in for requests.Response
    resp = MagicMock()  # an object that accepts any attribute
    resp.status_code = status  # HTTP status
    resp.text = text  # raw body text
    if body is None:  # no JSON body
        resp.json.side_effect = ValueError("not JSON")  # .json() fails like it does for HTML error pages
    else:  # a JSON body
        resp.json.return_value = body  # .json() returns it
    return resp  # the fake


@pytest.fixture(autouse=True)  # runs around every test in this file
def clear_cache():  # Streamlit's cache lives in this process; one test's fake data must not leak into the next
    st.cache_data.clear()  # empty before the test
    yield  # run the test
    st.cache_data.clear()  # empty after the test


# ---------- client tests (no network) ----------

def test_client_success():  # a 200 answer comes back as a dict
    with patch.object(api_client.requests, "get", return_value=fake_response(200, FAKE_HEALTH)):  # requests.get returns our fake
        assert api_client.get_health()["model_file"] == "lgbm_tuned.txt"  # the dict passed straight through


def test_client_404_keeps_api_message():  # an API error keeps its status and the API's own explanation
    body = {"detail": "No features for 2020-01-01. Available: 2025-10-07 to 2026-11-10"}  # FastAPI's error shape
    with patch.object(api_client.requests, "get", return_value=fake_response(404, body)):  # requests.get returns a 404
        with pytest.raises(ApiError) as err:  # the client must raise ApiError
            api_client.get_risk(dt.date(2020, 1, 1))  # ask for a day outside the window
    assert err.value.status == 404  # status kept
    assert "Available: 2025-10-07" in str(err.value)  # the API's message kept


def test_client_non_json_error():  # an error page that isn't JSON (e.g. from a proxy) still becomes a clean ApiError
    with patch.object(api_client.requests, "get", return_value=fake_response(502, None, "<html>Bad Gateway</html>")):  # HTML body
        with pytest.raises(ApiError) as err:  # must not crash on .json()
            api_client.get_health()  # any call
    assert err.value.status == 502  # status kept
    assert "Bad Gateway" in str(err.value)  # the raw text used as the message


@pytest.mark.parametrize("exc, words", [  # two network failures and the words their messages must contain
    (requests.ConnectionError("refused"), "Can't reach"),  # nothing listening
    (requests.Timeout("slow"), "took too long"),  # no answer in time
])  # end of the cases
def test_client_network_errors(exc, words):  # network failures become ApiError with no status code
    with patch.object(api_client.requests, "get", side_effect=exc):  # requests.get raises instead of answering
        with pytest.raises(ApiError) as err:  # turned into ApiError
            api_client.get_health()  # any call
    assert err.value.status is None  # no HTTP status, because there was no answer
    assert words in str(err.value)  # friendly message


def test_client_explain_params():  # get_explain sends the right URL and query parameters
    with patch.object(api_client.requests, "get", return_value=fake_response(200, fake_explain(7))) as get:  # keep a handle on the fake
        api_client.get_explain(7, dt.date(2026, 8, 10), k=3)  # cell 7 on a given day
    url, kwargs = get.call_args.args[0], get.call_args.kwargs  # what the client actually sent
    assert url.endswith("/explain/7")  # cell in the path
    assert kwargs["params"] == {"k": 3, "day": "2026-08-10"}  # k and day in the query string


# ---------- page tests (Streamlit AppTest, fake API) ----------

def run_page(health=FAKE_HEALTH, risk=None, health_error=None) -> AppTest:  # run the page once with fake client functions
    risk = risk or fake_risk(TODAY)  # default: fresh data (forecast for today)
    with patch("app.api_client.get_health", side_effect=health_error, return_value=health), \
         patch("app.api_client.get_risk", return_value=risk), \
         patch("app.api_client.get_explain", side_effect=fake_explain):  # the page imports these names on every run
        at = AppTest.from_file(APP_FILE, default_timeout=30)  # load the script (30 s: first import of pydeck/pandas is slow)
        at.run()  # run it top to bottom, like one page load
    return at  # the rendered page, ready to inspect


def metric_values(at: AppTest) -> dict:  # all cards as {label: value}
    return {m.label: m.value for m in at.metric}  # label text -> shown value


def test_page_loads_with_cards_and_panel():  # normal page: no crash, right cards, panel explains the riskiest cell
    at = run_page()  # fresh fake data
    assert not at.exception  # the script ran without an uncaught error
    assert at.title[0].value == "🔥 Lebanon Fire Radar"  # page heading
    cards = metric_values(at)  # every card on the page
    assert cards["🟠 High"] == "1"  # one high cell
    assert cards["🔴 Extreme"] == "0"  # no extreme cells
    assert cards["Highest risk"] == "2.00%"  # cell 2's risk, formatted
    assert any("### Cell 2" in m.value for m in at.markdown)  # the panel defaults to the riskiest cell (cell 2)
    assert any("0.0039" in m.value for m in at.markdown)  # small values are not rounded to 0.00 (format_value fix)
    assert len(at.warning) == 0  # fresh data: no stale banner


def test_page_api_down_shows_error_and_retry():  # API unreachable: friendly red box + Retry, no traceback
    at = run_page(health_error=ApiError("Can't reach the API at http://test:8000. Is it running?"))  # /health fails
    assert not at.exception  # handled, not crashed
    assert "Can't reach the API" in at.error[0].value  # the red box shows the client's message
    assert at.button[0].label == "🔄 Retry"  # the Retry button is there
    assert len(at.metric) == 0  # st.stop(): no cards drawn without data


def test_page_stale_data_banner():  # data older than today: yellow banner warns the visitor
    at = run_page(risk=fake_risk(TODAY - dt.timedelta(days=2)))  # newest forecast is 2 days old
    assert not at.exception  # the page still works
    assert any("2 days old" in w.value for w in at.warning)  # the banner says how old