"""Request and response shapes for the wildfire-risk API."""  # file description

import datetime as dt  # date types, under the short name dt (so the type never clashes with fields called "date")
from enum import Enum  # fixed list of allowed values

from pydantic import BaseModel, Field  # BaseModel = typed data class with validation, Field = extra rules and docs


class RiskLevel(str, Enum):  # the four levels from models/risk_levels.json (str: shown as text in JSON)
    low = "low"  # bottom 80% of cell-days
    moderate = "moderate"  # top 20%
    high = "high"  # top 5%
    extreme = "extreme"  # top 1%


class CellFeatures(BaseModel):  # one grid cell on one day: the input for one prediction
    cell_id: int = Field(..., ge=0, description="Grid cell ID from lebanon_grid.gpkg")  # required (...), must be 0 or more
    date: dt.date = Field(..., description="Day t: the day the features describe")  # required; the forecast is for the next day
    features: dict[str, float | None] = Field(  # feature name -> value; None means missing (LightGBM handles it)
        ..., description="All model features by name, as listed in model_meta.json"  # required, with a note for /docs
    )  # end of the features field


class BatchRequest(BaseModel):  # many cells in one call (used by the daily pipeline)
    rows: list[CellFeatures] = Field(..., min_length=1, max_length=5000)  # at least 1 row, at most 5,000 (10 days of all 488 cells)


class Reason(BaseModel):  # one SHAP reason behind a prediction (Step 10)
    feature: str  # the feature's code name, e.g. "vpd_max"
    label: str  # plain-English name, e.g. "air dryness (VPD)"
    value: float | None  # the feature's value for this cell-day
    impact: float  # SHAP push on the log-odds (+ raises risk, - lowers it)
    direction: str  # "raises risk" or "lowers risk"


class Prediction(BaseModel):  # the answer for one cell-day
    cell_id: int  # which cell
    date: dt.date  # day t (features day)
    forecast_for: dt.date  # day t+1 (the day the risk is about)
    risk: float = Field(..., ge=0, le=1)  # fire probability from the model, between 0 and 1
    level: RiskLevel  # low / moderate / high / extreme
    reasons: list[Reason] | None = None  # top SHAP reasons, only when asked for (None otherwise)


class BatchResponse(BaseModel):  # the answer for a batch
    count: int  # number of predictions returned
    predictions: list[Prediction]  # one Prediction per input row, in the same order


class HealthResponse(BaseModel):  # the answer of /health
    status: str  # "ok" when the model is loaded
    model_file: str  # which model file is serving
    trained_to: str  # last day in the training data
    n_features: int  # number of features the model expects
    risk_cutoffs: dict[str, float]  # minimum probability for moderate / high / extreme


class RiskMapResponse(BaseModel):  # the answer of /risk/{date}: every cell, ready for the map
    date: dt.date  # day t
    forecast_for: dt.date  # day t+1
    n_cells: int  # number of cells scored (488)
    level_counts: dict[str, int]  # how many cells are at each level
    geojson: dict  # GeoJSON FeatureCollection: one polygon per cell with its risk and level