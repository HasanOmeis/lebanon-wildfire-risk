"""Small client for the wildfire API: every call the Streamlit map makes goes through here."""  # file description

import datetime as dt  # dates for /risk/{day} and /explain
import os  # reads the API address from an environment variable

import requests  # HTTP calls to the FastAPI service

API_URL = os.getenv("WILDFIRE_API_URL", "http://127.0.0.1:8000").rstrip("/")  # API address; local by default, set on Day 17 for the cloud
TIMEOUT = (5, 30)  # (seconds to connect, seconds to wait for the answer); the full map takes a moment to build


class ApiError(Exception):  # one error type for everything that can go wrong, so the page only catches this
    def __init__(self, message: str, status: int | None = None):  # message for the user, HTTP status if there was one
        super().__init__(message)  # normal Exception behaviour (str(err) gives the message)
        self.status = status  # e.g. 404 when a day isn't served; None when the API couldn't be reached


def _get(path: str, params: dict | None = None) -> dict:  # shared GET: builds the URL, handles every failure the same way
    url = f"{API_URL}{path}"  # full address, e.g. http://127.0.0.1:8000/risk/latest
    try:  # the request itself can fail before any answer arrives
        resp = requests.get(url, params=params, timeout=TIMEOUT)  # send it
    except requests.Timeout as err:  # connected but no answer in time, or couldn't connect in time
        raise ApiError(f"The API at {API_URL} took too long to answer.") from err  # friendly message, original error kept
    except requests.ConnectionError as err:  # nothing listening at that address (container stopped, wrong URL)
        raise ApiError(f"Can't reach the API at {API_URL}. Is it running?") from err  # friendly message, original error kept
    if resp.status_code != 200:  # the API answered, but with an error
        try:  # FastAPI errors are JSON like {"detail": "..."}
            detail = resp.json().get("detail", resp.text)  # the API's own explanation
        except ValueError:  # not JSON (e.g. a proxy's HTML error page)
            detail = resp.text[:200]  # first part of the raw text
        raise ApiError(str(detail), status=resp.status_code)  # pass the API's message and status code on
    return resp.json()  # success: the answer as Python dicts and lists


def get_health() -> dict:  # GET /health: model file, training end date, risk cutoffs
    return _get("/health")  # no parameters needed


def get_risk(day: dt.date | None = None) -> dict:  # GET /risk/latest or /risk/{day}: every cell's risk plus the polygons
    path = "/risk/latest" if day is None else f"/risk/{day.isoformat()}"  # no day = newest served day
    return _get(path)  # date, forecast_for, n_cells, level_counts, geojson


def get_explain(cell_id: int, day: dt.date | None = None, k: int = 3) -> dict:  # GET /explain/{cell_id}: why one cell has its risk
    params = {"k": k}  # how many SHAP reasons to return
    if day is not None:  # a specific day was asked for
        params["day"] = day.isoformat()  # ?day=YYYY-MM-DD
    return _get(f"/explain/{cell_id}", params)  # cell_id, date, forecast_for, risk, level, reasons