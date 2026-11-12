"""FastAPI app: serves next-day wildfire risk for Lebanon's 5 km grid."""  # file description

import datetime as dt  # date type for URLs, and timedelta for the forecast day (t + 1)
import hmac  # compares secrets in constant time
import logging  # writes log lines
import os  # reads RELOAD_TOKEN from the environment
import time  # measures how long each request takes
from contextlib import asynccontextmanager  # turns a function into "run at startup, then at shutdown"

import pandas as pd  # type of the feature rows passed between helpers
from fastapi import Depends, FastAPI, Header, HTTPException, Path, Query, Request  # Header = read a value from the request headers
from fastapi.responses import JSONResponse, RedirectResponse  # JSONResponse = build an answer by hand, RedirectResponse = send elsewhere

from .predictor import FeatureError, Predictor  # our model loader and its "bad features" error (Step 5)
from .schemas import (  # request and response shapes (Step 4)
    BatchRequest,  # body of /predict/batch
    BatchResponse,  # answer of /predict/batch
    CellFeatures,  # body of /predict
    HealthResponse,  # answer of /health
    Prediction,  # answer for one cell-day (also used by /explain)
    RiskMapResponse,  # answer of /risk
)  # end of the imports
from .hub_sync import sync_from_hub  # downloads the model + serving files from Hugging Face when HF_REPO is set (Day 17)
from .store import FeatureStore  # serving data loader (Step 9B)

logging.basicConfig(  # set up logging once, when the app is imported
    level=logging.INFO,  # show INFO and above (INFO, WARNING, ERROR)
    format="%(asctime)s %(levelname)-7s %(name)s | %(message)s",  # time, level (padded to 7 letters), logger name, message
)  # end of the setup
log = logging.getLogger("wildfire.api")  # this app's own logger, so its lines are easy to spot


def load_state(app: FastAPI) -> None:  # sync from Hugging Face (cloud only), load model + serving data, then swap them in together
    commit = sync_from_hub()  # cloud: download the newest files into models/ and data/serving/; laptop: returns None and does nothing (Day 17)
    predictor = Predictor()  # load the model, metadata and cutoffs; a broken file raises here
    store = FeatureStore()  # load the serving features and grid; a broken file raises here
    app.state.predictor, app.state.store = predictor, store  # swap both only after BOTH loaded fine
    app.state.hub_commit = commit  # which Hugging Face upload is serving (None = local files) (Day 17)
    log.info("Model %s loaded: trained to %s, %d features", predictor.model_file, predictor.trained_to, len(predictor.feature_cols))  # what's serving
    log.info("Serving features %s to %s (%d days)", store.first, store.last, len(store.by_day))  # the date window
    log.info("Files from: %s", f"Hugging Face commit {commit[:7]}" if commit else "local folders")  # where the files came from (Day 17)


@asynccontextmanager  # makes lifespan usable by FastAPI as a startup/shutdown block
async def lifespan(app: FastAPI):  # FastAPI calls this once when the server starts
    load_state(app)  # same loader as /reload, so startup and reload can never differ
    yield  # the server runs and handles requests while we're paused here
    log.info("Shutting down")  # runs when the server stops


app = FastAPI(  # create the web app
    title="Lebanon Wildfire-Risk API",  # shown at the top of /docs
    version="1.2.0",  # minor version bump: files can come from the Hugging Face Hub (Day 17)
    description="Next-day wildfire risk for every ~5 km grid cell in Lebanon (LightGBM + SHAP reasons).",  # short summary in /docs
    lifespan=lifespan,  # use our startup block
)  # end of the app


@app.middleware("http")  # runs around EVERY request: before the endpoint and after it
async def log_and_protect(request: Request, call_next):  # call_next = "run the endpoint and give me its answer"
    start = time.perf_counter()  # precise timer start
    path = request.url.path + (f"?{request.url.query}" if request.url.query else "")  # the path, plus ?options if any
    try:  # run the endpoint, ready for anything unexpected
        response = await call_next(request)  # the endpoint's answer (404s and 422s come back here as normal answers)
    except Exception:  # an error nobody planned for (a bug, a corrupt file...)
        ms = (time.perf_counter() - start) * 1000  # time spent, in milliseconds
        log.exception("%s %s -> 500 (%.0f ms) unexpected error", request.method, path, ms)  # ERROR line + full traceback, in OUR logs only
        return JSONResponse(status_code=500, content={"detail": "Internal server error"})  # the caller gets a clean answer, no internals
    ms = (time.perf_counter() - start) * 1000  # time spent, in milliseconds
    response.headers["X-Process-Time-ms"] = f"{ms:.1f}"  # tell the caller how long we took
    level = logging.INFO if response.status_code < 400 else logging.WARNING  # 4xx = the caller's mistake: worth a warning
    log.log(level, "%s %s -> %d (%.0f ms)", request.method, path, response.status_code, ms)  # one line per request
    return response  # send the answer on


def get_predictor(request: Request) -> Predictor:  # dependency: hands the loaded predictor to any endpoint that asks
    return request.app.state.predictor  # the current one (replaced by /reload)


def get_store(request: Request) -> FeatureStore:  # dependency: hands the loaded serving data to any endpoint that asks
    return request.app.state.store  # the current one (replaced by /reload)


def to_predictions(cells: list[CellFeatures], p: Predictor, reasons: bool) -> list[Prediction]:  # shared by /predict and /predict/batch
    try:  # the predictor may reject the features
        results = p.predict_rows([c.features for c in cells], with_reasons=reasons)  # score every row in one model call
    except FeatureError as err:  # missing, unknown or empty features
        raise HTTPException(status_code=422, detail=str(err)) from err  # the caller's mistake: 422 with the clear message, not a 500 crash
    return [  # one Prediction per input row, same order
        Prediction(  # build one answer
            cell_id=c.cell_id,  # echo the cell
            date=c.date,  # day t
            forecast_for=c.date + dt.timedelta(days=1),  # day t + 1
            **res,  # risk, level and reasons from the predictor
        )  # end of one answer
        for c, res in zip(cells, results)  # pair each input row with its result
    ]  # end of the list


def day_rows(day: dt.date, store: FeatureStore) -> pd.DataFrame:  # one day's rows, or a 404 if the day isn't served
    rows = store.rows_for(day)  # that day's 488 feature rows
    if rows is None:  # the day isn't in the serving window
        raise HTTPException(  # send a 404 Not Found
            status_code=404,  # the resource (this day) doesn't exist
            detail=f"No features for {day}. Available: {store.first} to {store.last}",  # tell the caller what does exist
        )  # end of the error
    return rows  # the day's rows


def build_map(day: dt.date, p: Predictor, store: FeatureStore) -> RiskMapResponse:  # score every cell on one day and attach it to the polygons
    rows = day_rows(day, store)  # that day's rows (or a 404)
    results = p.predict_frame(rows)  # risk and level for every cell, in row order
    by_cell = {int(c): r for c, r in zip(rows["cell_id"], results)}  # {cell_id: result}, to look up by cell
    features = []  # GeoJSON features (one per cell) will go here
    for cell in store.grid["features"]:  # each polygon from grid.geojson
        props = cell["properties"]  # cell_id, land_frac, lat, lon
        result = by_cell.get(int(props["cell_id"]))  # this cell's prediction
        if result is None:  # a polygon without features that day (shouldn't happen)
            continue  # skip it rather than crash
        features.append({  # one map feature
            "type": "Feature",  # GeoJSON requires this
            "geometry": cell["geometry"],  # the polygon shape
            "properties": {**props, "risk": result["risk"], "level": result["level"]},  # the cell's info plus its risk
        })  # end of one feature
    levels = ["low", *p.cutoffs]  # low, moderate, high, extreme
    counts = {lvl: sum(f["properties"]["level"] == lvl for f in features) for lvl in levels}  # cells at each level (zeros included)
    return RiskMapResponse(  # build the answer
        date=day,  # day t
        forecast_for=day + dt.timedelta(days=1),  # day t + 1
        n_cells=len(features),  # 488
        level_counts=counts,  # summary for the map legend
        geojson={"type": "FeatureCollection", "features": features},  # the whole map, ready to draw
    )  # end of the answer


@app.get("/", include_in_schema=False)  # the bare address; hidden from /docs
def root() -> RedirectResponse:  # opening http://127.0.0.1:8000 in a browser...
    return RedirectResponse(url="/docs")  # ...goes straight to the interactive docs


@app.get("/health", response_model=HealthResponse, tags=["system"])  # GET /health, answer shaped like HealthResponse, grouped under "system"
def health(p: Predictor = Depends(get_predictor)) -> HealthResponse:  # FastAPI fills p by calling get_predictor
    return HealthResponse(  # build the answer
        status="ok",  # we only get here if the model loaded
        model_file=p.model_file,  # which model is serving
        trained_to=p.trained_to,  # how fresh it is
        n_features=len(p.feature_cols),  # 52
        risk_cutoffs=p.cutoffs,  # the level cutoffs in use
    )  # end of the answer


@app.post("/reload", tags=["system"])  # POST /reload, called by the daily pipeline after it publishes
def reload(  # sync + reload model and serving data without restarting the server
    request: Request,  # gives access to app.state
    x_reload_token: str | None = Header(None, description="Secret from RELOAD_TOKEN"),  # read from the X-Reload-Token header
) -> dict:  # small JSON answer
    expected = os.getenv("RELOAD_TOKEN", "")  # the secret this server was started with
    if not expected:  # no secret configured
        raise HTTPException(status_code=503, detail="Reload is disabled (RELOAD_TOKEN not set)")  # switched off, safely
    if not x_reload_token or not hmac.compare_digest(x_reload_token, expected):  # missing or wrong token
        raise HTTPException(status_code=401, detail="Invalid or missing X-Reload-Token")  # refused
    old_last = request.app.state.store.last  # newest served day before the reload
    try:  # loading new files can fail
        load_state(request.app)  # sync, load both, then swap (old data stays in memory if this raises)
    except Exception as err:  # Hugging Face unreachable, or broken files
        log.exception("Reload failed; still serving the previous data")  # full details in OUR logs
        raise HTTPException(status_code=500, detail="Reload failed; still serving the previous data") from err  # clean answer
    store, p = request.app.state.store, request.app.state.predictor  # the freshly loaded objects
    return {  # what changed
        "status": "reloaded",  # success
        "served_from": str(store.first),  # oldest served day
        "served_to": str(store.last),  # newest served day
        "previous_served_to": str(old_last),  # newest served day before
        "forecast_for": str(store.last + dt.timedelta(days=1)),  # the day /risk/latest is now about
        "model_file": p.model_file,  # which model is serving
        "hub_commit": request.app.state.hub_commit,  # which Hugging Face upload is now serving (None = local files) (Day 17)
    }  # end of the answer


@app.post("/predict", response_model=Prediction, tags=["predict"])  # POST /predict: features go in the request body
def predict(  # one cell-day in, one prediction out
    cell: CellFeatures,  # the JSON body, already checked by Pydantic
    reasons: bool = Query(False, description="Also return the top 3 SHAP reasons"),  # optional ?reasons=true in the URL
    p: Predictor = Depends(get_predictor),  # the loaded model
) -> Prediction:  # answer shaped like Prediction
    return to_predictions([cell], p, reasons)[0]  # a batch of one, take its only answer


@app.post("/predict/batch", response_model=BatchResponse, tags=["predict"])  # POST /predict/batch: many rows in one body
def predict_batch(  # many cell-days in, the same number of predictions out
    body: BatchRequest,  # {"rows": [...]}, 1 to 5,000 rows, already checked by Pydantic
    reasons: bool = Query(False, description="Also return the top 3 SHAP reasons for every row"),  # optional, slower for big batches
    p: Predictor = Depends(get_predictor),  # the loaded model
) -> BatchResponse:  # answer shaped like BatchResponse
    preds = to_predictions(body.rows, p, reasons)  # score all rows together
    return BatchResponse(count=len(preds), predictions=preds)  # the count plus every prediction


@app.get("/risk/latest", response_model=RiskMapResponse, tags=["map"])  # GET /risk/latest: must come BEFORE /risk/{day}
def risk_latest(  # the newest day in the serving data
    p: Predictor = Depends(get_predictor),  # the loaded model
    store: FeatureStore = Depends(get_store),  # the loaded serving data
) -> RiskMapResponse:  # answer shaped like RiskMapResponse
    return build_map(store.last, p, store)  # map for the newest day


@app.get("/risk/{day}", response_model=RiskMapResponse, tags=["map"])  # GET /risk/2025-10-01: any day in the serving window
def risk_on_day(  # the map for one chosen day
    day: dt.date,  # taken from the URL; FastAPI turns "2025-10-01" into a date or answers 422 if it isn't one
    p: Predictor = Depends(get_predictor),  # the loaded model
    store: FeatureStore = Depends(get_store),  # the loaded serving data
) -> RiskMapResponse:  # answer shaped like RiskMapResponse
    return build_map(day, p, store)  # map for that day


@app.get("/explain/{cell_id}", response_model=Prediction, tags=["explain"])  # GET /explain/464?day=...&k=...
def explain(  # why one cell has its risk on one day
    cell_id: int = Path(..., ge=0, description="Grid cell ID (0-487)"),  # from the URL path; negative numbers get an automatic 422
    day: dt.date | None = Query(None, description="Day t (default: the latest served day)"),  # optional ?day=YYYY-MM-DD
    k: int = Query(3, ge=1, le=10, description="How many reasons to return"),  # optional ?k=5, between 1 and 10
    p: Predictor = Depends(get_predictor),  # the loaded model
    store: FeatureStore = Depends(get_store),  # the loaded serving data
) -> Prediction:  # same shape as /predict, with reasons filled in
    day = day or store.last  # no day given: use the newest
    day_rows(day, store)  # 404 if the day isn't served (the rows themselves aren't needed here)
    row = store.cell_row(day, cell_id)  # this cell's features on that day
    if row is None:  # the cell isn't in the grid
        raise HTTPException(status_code=404, detail=f"Cell {cell_id} not found. Valid cells: 0 to 487")  # a different 404 message for a bad cell
    result = p.predict_frame(row, with_reasons=True, k=k)[0]  # score the one row with its top k reasons
    return Prediction(  # build the answer
        cell_id=cell_id,  # the cell asked for
        date=day,  # day t
        forecast_for=day + dt.timedelta(days=1),  # day t + 1
        **result,  # risk, level and reasons
    )  # end of the answer