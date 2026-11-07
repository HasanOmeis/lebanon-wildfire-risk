"""Loads the production model once and turns features into risk, levels and reasons."""  # file description

import json  # reads model_meta.json and risk_levels.json
from pathlib import Path  # file paths that work on Windows and Linux

import lightgbm as lgb  # loads the saved trees
import numpy as np  # sorting the strongest reasons
import pandas as pd  # model inputs as tables

ROOT = Path(__file__).resolve().parents[2]  # project root (src/api/predictor.py -> two folders up)
MODEL_DIR = ROOT / "models"  # where lgbm_tuned.txt, model_meta.json and risk_levels.json live

FEATURE_LABELS = {  # plain-English names for all 52 features, shown in reasons and on the map
    # fire activity in this cell
    "n_detections": "fire detections here today",  # number of satellite fire points in the cell today
    "frp_sum": "fire intensity here today",  # total fire radiative power (MW) of today's detections
    "fire_today": "fire detected here today",  # 1 if the cell burned today
    "fires_last_7d": "fire days here this week",  # fire days in the cell over the last 7 days
    "fires_last_30d": "fire days here this month",  # fire days over the last 30 days
    "fires_last_365d": "fire days here this year",  # fire days over the last 365 days
    "days_since_fire": "days since the last fire here",  # recency of fire in the cell
    "cell_fire_rate": "how often this area burns",  # long-run share of days the cell burned
    # fire activity next door
    "nbr_fires_today": "fires in neighboring cells today",  # touching cells that burned today
    "nbr_fires_last_7d": "fires in neighboring cells this week",  # their fire days over the last 7 days
    "n_neighbors": "number of neighboring cells",  # 8 inland, fewer on the coast and borders
    # today's weather
    "tmax": "today's max temperature",  # daily maximum temperature
    "rh_min": "today's minimum humidity",  # lowest relative humidity of the day
    "precip": "rain today",  # today's rainfall
    "wind_max": "today's max wind speed",  # strongest sustained wind
    "gust_max": "today's strongest gust",  # strongest wind gust
    "et0": "today's evaporation demand",  # reference evapotranspiration: how fast the air dries plants
    "vpd_max": "today's air dryness (VPD)",  # vapor pressure deficit
    # recent weather windows
    "tmax_mean_3d": "heat over the last 3 days",  # 3-day average max temperature
    "rh_min_mean_3d": "humidity over the last 3 days",  # 3-day average minimum humidity
    "vpd_max_mean_3d": "air dryness over the last 3 days",  # 3-day average VPD
    "wind_max_3d": "strongest wind in the last 3 days",  # 3-day max wind
    "tmax_mean_7d": "heat over the last week",  # 7-day average max temperature
    "rh_min_mean_7d": "humidity over the last week",  # 7-day average minimum humidity
    "vpd_max_mean_7d": "air dryness over the last week",  # 7-day average VPD
    "wind_max_7d": "strongest wind in the last week",  # 7-day max wind
    "precip_sum_7d": "rain over the last week",  # 7-day rainfall total
    "precip_sum_30d": "rain over the last month",  # 30-day rainfall total
    "precip_sum_90d": "rain over the last 3 months",  # 90-day rainfall total
    "et0_sum_30d": "evaporation demand over the last month",  # 30-day total evapotranspiration
    "dry_streak": "days without rain",  # consecutive dry days
    # weather compared with normal
    "tmax_anom": "heat compared with normal",  # this week's heat minus the cell's usual for this month
    "rh_min_anom": "humidity compared with normal",  # humidity minus the usual
    "vpd_max_anom": "air dryness compared with normal",  # VPD minus the usual
    "et0_anom": "evaporation demand compared with normal",  # evapotranspiration minus the usual
    # vegetation
    "ndvi": "vegetation (fuel) amount",  # greenness from satellite
    "ndvi_age_days": "age of the vegetation image",  # days since the latest satellite composite
    "ndvi_anomaly": "vegetation compared with normal",  # greenness minus the usual for this time of year
    "ndvi_change_30d": "vegetation change over 30 days",  # drying or greening trend
    # terrain
    "elev_mean": "elevation",  # average height of the cell
    "elev_max": "highest point",  # maximum height in the cell
    "elev_std": "terrain roughness",  # how much the height varies inside the cell
    "slope_mean": "average slope",  # mean steepness
    "slope_max": "steepest slope",  # maximum steepness
    "northness_mean": "north-facing slopes",  # how much the slopes face north (cooler, moister)
    "eastness_mean": "east-facing slopes",  # how much the slopes face east
    "south_steep_frac": "steep south-facing slopes",  # share of steep slopes facing the sun (hot, dry)
    "land_frac": "share of the cell on land",  # coastal cells are partly sea
    # calendar
    "month": "time of year",  # calendar month
    "doy_sin": "season (cycle part 1)",  # day of year as a smooth cycle, sine part
    "doy_cos": "season (cycle part 2)",  # day of year as a smooth cycle, cosine part
    "day_of_week": "day of the week",  # human activity (burning, picnics) changes with the week
}  # end of the labels


class FeatureError(ValueError):  # our own error type: the request's features don't fit the model
    """Raised when a request's features don't match what the model expects."""  # the API turns this into a 422 response


class Predictor:  # holds the model and everything needed to use it
    def __init__(self, model_dir: Path = MODEL_DIR):  # runs once, at API startup
        meta = json.loads((model_dir / "model_meta.json").read_text())  # model info saved by train_final.py
        self.model_file = meta["model_file"]  # "lgbm_tuned.txt"
        self.trained_to = meta["trained_to"]  # last training day, shown by /health
        self.model = lgb.Booster(model_file=str(model_dir / self.model_file))  # load the trees
        self.feature_cols = self.model.feature_name()  # the exact column order the model was trained on
        if self.feature_cols != meta["feature_cols"]:  # the metadata must describe this very model
            raise RuntimeError("model_meta.json feature list does not match the model file")  # stop startup: files are out of sync
        self.categories = dict(zip(meta["categorical"], self.model.pandas_categorical))  # {"month": [1..12], "day_of_week": [0..6]} from training
        levels = json.loads((model_dir / "risk_levels.json").read_text())  # cutoffs from Step 2
        self.cutoffs = levels["cutoffs"]  # {"moderate": ..., "high": ..., "extreme": ...}, lowest first

    def check(self, features: dict) -> None:  # make sure one row has exactly the right features
        missing = [c for c in self.feature_cols if c not in features]  # features the model needs but the row lacks
        unknown = [c for c in features if c not in self.feature_cols]  # names the model doesn't know (often typos)
        empty_cat = [c for c in self.categories if c in features and features[c] is None]  # month/day_of_week can't be missing
        problems = []  # collect every problem, so the caller sees them all at once
        if missing:  # any missing features?
            problems.append(f"missing features: {missing}")  # list them
        if unknown:  # any unknown names?
            problems.append(f"unknown features: {unknown}")  # list them
        if empty_cat:  # any empty categorical columns?
            problems.append(f"these features need a value: {empty_cat}")  # list them
        if problems:  # something is wrong
            raise FeatureError("; ".join(problems))  # one clear message

    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:  # turn a table into model-ready inputs
        X = df[self.feature_cols].astype(float)  # right columns, right order, all numbers (None becomes NaN)
        for col, cats in self.categories.items():  # month and day_of_week
            X[col] = pd.Categorical(X[col].astype(int), categories=cats)  # same category type and list as in training
        return X  # ready for model.predict

    def level(self, risk: float) -> str:  # turn one probability into a level name
        name = "low"  # start at the lowest level
        for lvl, cut in self.cutoffs.items():  # go up through moderate, high, extreme
            if risk >= cut:  # risk reaches this level's cutoff
                name = lvl  # move up to this level
        return name  # the highest level reached

    def predict_frame(self, df: pd.DataFrame, with_reasons: bool = False, k: int = 3) -> list[dict]:  # score every row of a table
        X = self.prepare(df)  # model-ready inputs
        risk = self.model.predict(X)  # fire probability per row
        out = [{"risk": round(float(r), 6), "level": self.level(r), "reasons": None} for r in risk]  # one result per row
        if with_reasons:  # only do the extra work when asked
            pushes = self.model.predict(X, pred_contrib=True)[:, :-1]  # per-feature pushes (= SHAP, checked on Day 12), drop the base column
            for i, row in enumerate(out):  # each row's result
                top = np.argsort(-np.abs(pushes[i]))[:k]  # positions of the k strongest pushes, either direction
                reasons = []  # this row's reasons
                for j in top:  # each strong feature
                    name = self.feature_cols[j]  # its code name
                    value = df[name].iloc[i]  # its real value for this row
                    reasons.append({  # one reason
                        "feature": name,  # code name
                        "label": FEATURE_LABELS.get(name, name),  # plain-English name, or the code name if none
                        "value": None if pd.isna(value) else float(value),  # NaN can't go into JSON, so send None
                        "impact": round(float(pushes[i, j]), 3),  # push on the log-odds
                        "direction": "raises risk" if pushes[i, j] > 0 else "lowers risk",  # which way it pushed
                    })  # end of one reason
                row["reasons"] = reasons  # attach the reasons to this row's result
        return out  # same order as the input rows

    def predict_rows(self, rows: list[dict], with_reasons: bool = False) -> list[dict]:  # score feature dicts sent to the API
        for i, features in enumerate(rows):  # check every row before scoring any
            try:  # check this row
                self.check(features)  # raises FeatureError if something is wrong
            except FeatureError as err:  # it was wrong
                raise FeatureError(f"row {i}: {err}") from err  # add the row number so the caller can find it
        df = pd.DataFrame(rows, columns=self.feature_cols)  # one row per dict, columns in model order
        return self.predict_frame(df, with_reasons)  # score them


if __name__ == "__main__":  # quick self-test: python -m src.api.predictor
    p = Predictor()  # load everything, like the API will
    print(f"Model {p.model_file}, trained to {p.trained_to}, {len(p.feature_cols)} features")  # basic info
    print("Cutoffs:", p.cutoffs)  # should match Step 2
    df = pd.read_parquet(ROOT / "data" / "processed" / "features.parquet")  # the full feature table
    last = df[df["date"] == df["date"].max()]  # all cells on the latest day
    results = p.predict_frame(last)  # score the whole country for that day
    print(f"\nLatest day {pd.to_datetime(last['date'].iloc[0]).date()}: {len(results)} cells")  # day and cell count
    print(pd.Series([r["level"] for r in results]).value_counts())  # how many cells at each level
    rows = last[p.feature_cols].head(2).to_dict("records")  # two cells as plain dicts, like an API request
    for r in p.predict_rows(rows, with_reasons=True):  # score them with reasons
        print(f"\nrisk {r['risk']:.4f} -> {r['level']}")  # risk and level
        for reason in r["reasons"]:  # top 3 reasons
            print(f"  {reason['label']} = {reason['value']} -> {reason['direction']} ({reason['impact']:+.3f})")  # one line each
    try:  # now send a bad request on purpose
        p.predict_rows([{"tmax": 30.0, "vpd_mx": 2.0}])  # missing almost everything + a typo
    except FeatureError as err:  # it should be rejected
        print("\nRejected as expected:", str(err)[:160], "...")  # show the start of the message