import json  # writes the cutoffs file

import numpy as np  # computes the quantile cutoffs
import pandas as pd  # tables and binning

from explain import ROOT, TARGET, load_model, load_features, prepare_X  # Day 12 helpers: same model loading and input preparation

LEVELS_PATH = ROOT / "models" / "risk_levels.json"  # the file the API will read
START = "2024-01-01"  # set the cutoffs on the recent period (2024 onward)
QUANTILES = {"moderate": 0.80, "high": 0.95, "extreme": 0.99}  # risk must beat this share of all cell-days to reach the level
NAMES = ["low", "moderate", "high", "extreme"]  # level names, lowest to highest


if __name__ == "__main__":  # runs only when you call this file directly
    model = load_model()  # tuned production model (models/lgbm_tuned.txt)
    df = load_features()  # the full feature table
    df = df[pd.to_datetime(df["date"]) >= START].copy()  # keep 2024 onward, as a separate copy
    risk = model.predict(prepare_X(df, model))  # fire probability for every cell-day
    cutoffs = {lvl: float(np.quantile(risk, q)) for lvl, q in QUANTILES.items()}  # the probability at each quantile
    bins = [-np.inf, *cutoffs.values(), np.inf]  # edges: below moderate, moderate-high, high-extreme, above extreme
    df["level"] = pd.cut(risk, bins=bins, labels=NAMES, right=False)  # give every cell-day its level (cutoff counts as the higher level)

    check = df.groupby("level", observed=False).agg(  # one row per level
        share=(TARGET, "size"),  # how many cell-days fall in it
        fire_rate=(TARGET, "mean"),  # how often a fire really followed the next day
    )  # end of the summary
    check["share"] = check["share"] / len(df)  # turn counts into shares of all cell-days
    print(check.round(4))  # fire_rate should rise from low to extreme

    out = {  # everything the API needs to label a probability
        "cutoffs": {k: round(v, 6) for k, v in cutoffs.items()},  # minimum probability for each level
        "quantiles": QUANTILES,  # how the cutoffs were defined
        "levels": NAMES,  # names in order, lowest first
        "based_on": f"{df['date'].min():%Y-%m-%d} to {df['date'].max():%Y-%m-%d}, {len(df):,} cell-days",  # data used
        "fire_rate_by_level": check["fire_rate"].round(4).to_dict(),  # real fire rate per level, for the README
    }  # end of the file contents
    LEVELS_PATH.write_text(json.dumps(out, indent=2))  # save with readable indentation
    print(json.dumps(out["cutoffs"], indent=2))  # show the cutoffs
    print("Saved", LEVELS_PATH.relative_to(ROOT))  # confirmation