"""Tell the running API to reload its files after the pipeline publishes (never fails the run)."""  # file description

import logging  # log lines for the runner
import os  # reads API_URL and RELOAD_TOKEN (already loaded from .env by config)

import requests  # HTTP request to the API

from . import config as C  # noqa: F401  importing config runs load_dotenv, so the two variables are set

log = logging.getLogger("pipeline.api")  # this step's own logger


def reload_api() -> dict | None:  # returns the API's answer, or None if it couldn't be reached
    url = os.getenv("API_URL", "").rstrip("/")  # e.g. http://127.0.0.1:8000 (no trailing slash)
    token = os.getenv("RELOAD_TOKEN", "")  # the shared secret
    if not url or not token:  # not configured: nothing to notify
        log.info("API_URL or RELOAD_TOKEN not set; skipping API reload")  # record the skip
        return None  # nothing done
    try:  # the API may be stopped, starting, or unreachable
        resp = requests.post(f"{url}/reload", headers={"X-Reload-Token": token}, timeout=60)  # ask it to reload
    except requests.RequestException as e:  # connection refused, timeout, DNS...
        log.warning("API not reachable at %s (%s); it will load the new files on its next start", url, type(e).__name__)  # warn only
        return None  # the run still counts as OK
    if resp.status_code != 200:  # reachable but refused or failed
        log.warning("API reload returned %d: %s", resp.status_code, resp.text[:200])  # warn with the reason
        return None  # the run still counts as OK
    body = resp.json()  # {"status": "reloaded", "served_to": ..., ...}
    log.info("API reloaded: serving to %s (forecast for %s), model %s",  # success line...
             body["served_to"], body["forecast_for"], body["model_file"])  # ...with what it now serves
    return body  # handy for tests