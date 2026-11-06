from pathlib import Path   # handles file paths the same way on Windows and Linux

import lightgbm as lgb     # loads the tuned LightGBM model
import numpy as np         # numeric arrays for checks and sorting
import pandas as pd        # reads and writes tables
import shap                # computes SHAP explanations

ROOT = Path(__file__).resolve().parents[1]                              # project root folder (one level above src)
MODEL_PATH = ROOT / "models" / "lgbm_tuned.txt"                         # tuned model saved on Day 11
FEATURES_PATH = ROOT / "data" / "processed" / "features.parquet"        # final feature table from Day 7
TEST_PRED_PATH = ROOT / "data" / "processed" / "test_predictions.parquet"   # test-period predictions from Day 10
SAMPLE_PATH = ROOT / "data" / "processed" / "shap_sample.parquet"       # where the explained rows are saved
SHAP_PATH = ROOT / "data" / "processed" / "shap_values.parquet"         # where the SHAP values are saved
TARGET = "fire_tomorrow"                                                # label column: 1 if the cell burns the next day
CAT_COLS = ["month", "day_of_week"]                                     # columns the model treats as categories (features 48 and 51)
N_POS = 2000                                                            # max fire rows to keep in the sample
N_NEG = 3000                                                            # no-fire rows to keep in the sample
SEED = 42                                                               # fixed random seed so the sample is repeatable
FEATURE_LABELS = {                                                      # plain-English names for the features people will see on the map
    "cell_fire_rate": "how often this area burns",                     # long-run fire frequency of the cell
    "days_since_fire": "days since the last fire here",                # recency of fire in the cell
    "fire_today": "fire detected here today",                          # active fire in the cell today
    "nbr_fires_today": "fires in neighboring cells today",             # active fires next door today
    "nbr_fires_last_7d": "fires in neighboring cells this week",       # recent fires next door
    "elev_mean": "elevation",                                          # average height of the cell
    "ndvi": "vegetation (fuel) amount",                                # greenness from satellite
    "ndvi_change_30d": "vegetation change over 30 days",               # drying or greening trend
    "vpd_max": "air dryness (VPD)",                                    # vapor pressure deficit
    "tmax": "today's max temperature",                                 # daily max temperature
    "tmax_mean_7d": "heat over the last week",                         # 7-day average max temperature
    "rh_min": "minimum humidity",                                      # lowest relative humidity today
    "precip": "rain today",                                            # today's rainfall
    "precip_sum_30d": "rain over the last 30 days",                    # month rainfall
    "dry_streak": "days without rain",                                 # consecutive dry days
    "wind_max": "max wind speed",                                      # strongest wind today
    "month": "time of year",                                           # calendar month
}                                                                       # end of the labels


def load_model() -> lgb.Booster:                                        # function that loads the saved model
    return lgb.Booster(model_file=str(MODEL_PATH))                      # read the LightGBM text file into a Booster


def load_features() -> pd.DataFrame:                                    # function that loads the feature table
    return pd.read_parquet(FEATURES_PATH)                               # read the parquet file into a DataFrame


def prepare_X(df: pd.DataFrame, model: lgb.Booster) -> pd.DataFrame:    # function that builds model-ready inputs
    X = df[model.feature_name()].copy()                                 # keep the 52 features in training order, as a copy
    for col, cats in zip(CAT_COLS, model.pandas_categorical):           # pair each categorical column with its training categories
        X[col] = pd.Categorical(X[col].astype(int), categories=cats)    # turn the column into the same category type used in training
    return X                                                            # return inputs the model accepts


def top_reasons(model: lgb.Booster, df: pd.DataFrame, k: int = 3) -> list[dict]:   # risk + top k reasons for every row
    X = prepare_X(df, model)                                            # model-ready inputs
    contrib = model.predict(X, pred_contrib=True)                       # per-feature pushes (same as SHAP), last column = base value
    names = model.feature_name()                                        # feature names in the same order as the columns
    pushes = contrib[:, :-1]                                            # drop the base value column, keep the 52 pushes
    risk = 1 / (1 + np.exp(-contrib.sum(axis=1)))                       # base + pushes = log-odds, turned into a probability
    results = []                                                        # one entry per row will go here
    for i in range(len(X)):                                             # go through each row
        top = np.argsort(-np.abs(pushes[i]))[:k]                        # positions of the k strongest pushes, either direction
        reasons = [                                                     # build one small dict per reason
            {                                                           # start of one reason
                "feature": names[j],                                    # the feature's code name
                "label": FEATURE_LABELS.get(names[j], names[j]),        # plain-English name, or the code name if none
                "value": float(df[names[j]].iloc[i]),                   # the feature's actual value for this row
                "impact": round(float(pushes[i, j]), 3),                # how much it moved the log-odds
                "direction": "raises risk" if pushes[i, j] > 0 else "lowers risk",   # which way it pushed
            }                                                           # end of one reason
            for j in top                                                # one reason for each of the top k features
        ]                                                               # end of the reasons list
        results.append({"risk": round(float(risk[i]), 4), "reasons": reasons})   # store this row's risk and reasons
    return results                                                      # list of {risk, reasons} in the same order as df


def test_period() -> tuple[pd.Timestamp, pd.Timestamp]:                 # function that finds the test date range
    dates = pd.read_parquet(TEST_PRED_PATH, columns=["date"])["date"]   # read only the date column of the test predictions
    dates = pd.to_datetime(dates)                                       # make sure the dates are real datetime values
    return dates.min(), dates.max()                                     # first and last day of the test period


def sample_rows(df: pd.DataFrame) -> pd.DataFrame:                      # function that builds the balanced sample
    start, end = test_period()                                          # get the test date range
    dates = pd.to_datetime(df["date"])                                  # convert the table's dates to datetime
    test = df[(dates >= start) & (dates <= end)]                        # keep only rows inside the test period
    pos = test[test[TARGET] == 1]                                       # rows where a fire happened the next day
    neg = test[test[TARGET] == 0]                                       # rows with no fire the next day
    pos = pos.sample(n=min(N_POS, len(pos)), random_state=SEED)         # take up to N_POS fire rows at random
    neg = neg.sample(n=min(N_NEG, len(neg)), random_state=SEED)         # take up to N_NEG no-fire rows at random
    out = pd.concat([pos, neg])                                         # stack the fire and no-fire rows together
    return out.sample(frac=1, random_state=SEED).reset_index(drop=True) # shuffle the rows and reset the index


def compute_shap(model: lgb.Booster, X: pd.DataFrame) -> shap.Explanation:   # function that runs SHAP on the rows
    explainer = shap.TreeExplainer(model)                               # fast exact explainer made for tree models
    return explainer(X)                                                 # SHAP values for every row and feature (log-odds)


if __name__ == "__main__":                                              # run this part only when the file is run directly
    model = load_model()                                                # load the tuned model
    cols = model.feature_name()                                         # the 52 features, in training order
    sample = sample_rows(load_features())                               # build the 4,377-row sample from the test period
    X = prepare_X(sample, model)                                        # model-ready inputs with the right category types
    expl = compute_shap(model, X)                                       # compute the SHAP values
    print("SHAP values shape:", expl.values.shape)                      # should be (rows, 52)
    base = float(np.ravel(expl.base_values)[0])                         # the model's average raw score (log-odds)
    print("Base value (log-odds):", round(base, 3))                     # show the starting point of every explanation
    raw = model.predict(X, raw_score=True)                              # the model's raw log-odds score for each row
    rebuilt = expl.values.sum(axis=1) + base                            # base value plus all feature pushes
    print("Max additivity error:", float(np.abs(raw - rebuilt).max()))  # should be tiny, close to 0
    sample.to_parquet(SAMPLE_PATH, index=False)                         # save the explained rows (ids, date, target, features)
    pd.DataFrame(expl.values, columns=cols).to_parquet(SHAP_PATH, index=False)   # save the SHAP values, one column per feature
    print("Saved:", SAMPLE_PATH.name, "and", SHAP_PATH.name)            # confirm both files were written
    contrib = model.predict(X, pred_contrib=True)[:, :-1]               # LightGBM's own pushes, without the base column
    print("pred_contrib vs SHAP max difference:", float(np.abs(contrib - expl.values).max()))   # should be close to 0: both methods agree
    demo = sample.loc[sample[TARGET] == 1].head(2)                      # two real fire rows to try the helper on
    for (_, row), res in zip(demo.iterrows(), top_reasons(model, demo)):   # pair each row with its result
        print(f"\nCell {row['cell_id']} on {pd.to_datetime(row['date']).date()}: risk {res['risk']:.2%}")   # header for this row
        for r in res["reasons"]:                                        # go through its top 3 reasons
            print(f"  {r['label']} = {r['value']:.2f} -> {r['direction']} ({r['impact']:+.2f})")   # one readable line per reason