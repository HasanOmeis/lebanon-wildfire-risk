"""Logging and the 500 safety net: run with  python -m pytest tests/test_errors.py -v"""  # file description

import logging  # log levels for caplog

from fastapi.testclient import TestClient  # calls the app in Python, no server needed

from src.api.main import app, get_predictor  # the app and the dependency we'll swap out
from tests.test_api import pytestmark  # noqa: F401  # reuse the same "skip if model/serving files are missing" rule


class BrokenPredictor:  # a fake predictor that always crashes, like a bug would
    cutoffs = {"moderate": 0.1, "high": 0.2, "extreme": 0.3}  # build_map reads this before it would fail

    def predict_frame(self, *args, **kwargs):  # any scoring call...
        raise RuntimeError("secret internal detail: C:/models/lgbm_tuned.txt is corrupt")  # ...fails with an internal message


def test_unexpected_error_gives_clean_500(caplog):  # crashes become a clean 500, and the detail stays in the logs
    app.dependency_overrides[get_predictor] = lambda: BrokenPredictor()  # every endpoint now gets the broken predictor
    try:  # make sure the swap is undone even if an assert fails
        with TestClient(app) as client, caplog.at_level(logging.ERROR, logger="wildfire.api"):  # start the app, capture ERROR logs
            r = client.get("/explain/226")  # this endpoint will hit the crash
        assert r.status_code == 500  # server error code
        assert r.json() == {"detail": "Internal server error"}  # the caller sees only a generic message
        assert "secret" not in r.text  # no internal detail leaked to the caller
        assert "secret internal detail" in caplog.text  # but the full error IS in our logs
        assert "GET /explain/226 -> 500" in caplog.text  # with the request that caused it
    finally:  # always runs
        app.dependency_overrides.clear()  # put the real predictor back for other tests


def test_requests_are_logged_and_timed(caplog):  # every request gets a log line and a timing header
    with TestClient(app) as client, caplog.at_level(logging.INFO, logger="wildfire.api"):  # start the app, capture INFO logs
        ok = client.get("/health")  # a normal request
        bad = client.get("/explain/9999")  # a 404 request
    assert float(ok.headers["X-Process-Time-ms"]) >= 0  # timing header present and numeric
    assert "GET /health -> 200" in caplog.text  # the success was logged
    assert any(  # the 404 was logged...
        rec.levelno == logging.WARNING and "/explain/9999 -> 404" in rec.getMessage()  # ...as a WARNING
        for rec in caplog.records  # among all captured log records
    )  # end of the check
    assert bad.status_code == 404  # and it really was a 404