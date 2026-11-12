"""Tell a running API (laptop or live) to reload its files after the pipeline publishes (never fails the run)."""  # file description

import logging  # log lines for the runner
import os  # reads the API address and RELOAD_TOKEN (already loaded from .env by config)

import requests  # HTTP request to the API

from . import config as C  # noqa: F401  importing config runs load_dotenv, so the variables are set

log = logging.getLogger("pipeline.api")  # this step's own logger


def reload_api(url_var: str = "API_URL") -> dict | None:  # url_var = which .env variable holds the address; returns the API's answer or None
    url = os.getenv(url_var, "").rstrip("/")  # API_URL = laptop (http://127.0.0.1:8000), LIVE_API_URL = Render (Day 17)
    token = os.getenv("RELOAD_TOKEN", "")  # the shared secret (the same value is set on Render)
    if not url or not token:  # not configured: nothing to notify
        log.info("%s or RELOAD_TOKEN not set; skipping that API's reload", url_var)  # record the skip, naming which one
        return None  # nothing done
    try:  # the API may be stopped, asleep (Render free plan), or unreachable
        resp = requests.post(f"{url}/reload", headers={"X-Reload-Token": token}, timeout=120)  # 120 s: a sleeping Render service needs ~1 min to wake
    except requests.RequestException as e:  # connection refused, timeout, DNS...
        log.warning("API not reachable at %s (%s); it will load the new files on its next start", url, type(e).__name__)  # warn only
        return None  # the run still counts as OK
    if resp.status_code != 200:  # reachable but refused or failed
        log.warning("API reload at %s returned %d: %s", url, resp.status_code, resp.text[:200])  # warn with the reason
        return None  # the run still counts as OK
    body = resp.json()  # {"status": "reloaded", "served_to": ..., "hub_commit": ...}
    log.info("API at %s reloaded: serving to %s (forecast for %s), model %s",  # success line...
             url, body["served_to"], body["forecast_for"], body["model_file"])  # ...with what it now serves
    return body  # handy for tests