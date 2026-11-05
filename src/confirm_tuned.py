import time  # measures how long each year takes

import lightgbm as lgb  # the gradient-boosted trees library
import mlflow  # experiment tracking
import numpy as np  # NaN placeholders
import pandas as pd  # tables

from train_baseline import ROOT, TARGET, TRACKING_URI, load_data  # project folder, label, MLflow location, data loader
from train_lgbm import CATEGORICAL, to_lgb_frame  # same LightGBM inputs as Days 9–11
from validate import (  # reuse Day 10's strict-scheme pieces so the exam is identical
    TEST_START,  # 2024-01-01: never touched here
    TIME_TEST_YEARS,  # 2018–2023
    N_SPATIAL_FOLDS,  # 5 region groups
    add_blocks,  # cell -> 25 km block
    block_fold_map,  # block -> region group (same seed as Day 10)
    score,  # Day 10's full metric set
)  # end of Day 10 imports
from best_params import load_best_params  # the tuned winner from configs/best_params.json

DAY10_PATH = ROOT / "reports" / "validation_spacetime.csv"  # Day 10 results with the Day 9 settings
OUT_PATH = ROOT / "reports" / "validation_spacetime_tuned.csv"  # today's results with the tuned settings
METRICS = ["pr_auc", "lift", "roc_auc", "recall_top5", "precision_top5", "brier_skill"]  # columns to compare


def fit_predict_tuned(train, test, feature_cols, settings):  # train with the tuned settings, score the test part
    params, num_rounds, weight_power = settings  # unpack the winner
    y = train[TARGET]  # training labels
    n_pos = int(y.sum())  # fires in this training part
    weight = ((len(y) - n_pos) / n_pos) ** weight_power  # fire weight, same formula as the search (≈ 1 here)
    dtrain = lgb.Dataset(to_lgb_frame(train, feature_cols), y, categorical_feature=CATEGORICAL)  # LightGBM data
    model = lgb.train({**params, "scale_pos_weight": weight}, dtrain, num_boost_round=num_rounds)  # fixed tree count
    return model.predict(to_lgb_frame(test, feature_cols))  # fire scores for the test part


def run_spacetime_tuned(df, feature_cols, settings):  # Day 10's strict scheme, with the tuned settings
    fold_of_row = df["block"].map(block_fold_map(df))  # same region groups as Day 10
    rows = []  # one dict of results per test year
    for year in TIME_TEST_YEARS:  # one result per test year
        start = time.time()  # start the timer for this year
        is_year = df["date"].dt.year == year  # rows in the test year
        is_past = df["date"] < f"{year - 1}-12-31"  # earlier years, with the 1-day gap
        test = df[is_year]  # the full test year, all 488 cells
        proba = pd.Series(np.nan, index=test.index)  # empty score slot for every test row
        for k in range(N_SPATIAL_FOLDS):  # one model per held-out region group
            train = df[is_past & (fold_of_row != k)]  # past years, other regions only
            part = test[fold_of_row[is_year] == k]  # this year's rows in the held-out regions
            proba.loc[part.index] = fit_predict_tuned(train, part, feature_cols, settings)  # fill in their scores
        s = score(test, proba.values)  # score the whole stitched year
        rows.append({"fold": f"spacetime_{year}", "test_fires": int(test[TARGET].sum()), **s})  # save the row
        print(f"{year}: pr_auc={s['pr_auc']:.4f}  recall_top5={s['recall_top5']:.4f}  "  # progress line
              f"brier_skill={s['brier_skill']:.2f}  ({time.time() - start:.0f} s)")  # second half of the line
    return pd.DataFrame(rows).set_index("fold")  # all years as one table


def mean_std(table):  # "mean ± std" text for each metric
    return {m: f"{table[m].mean():.4f} ± {table[m].std():.4f}" for m in METRICS}  # one string per metric


if __name__ == "__main__":  # runs only when you call this file directly
    df, feature_cols = load_data()  # same table and features as Days 8–11
    df = df[df["date"] < TEST_START]  # 2024+ stays locked
    df = add_blocks(df)  # attach the 25 km block id to every row
    settings = load_best_params()  # the tuned winner
    tuned = run_spacetime_tuned(df, feature_cols, settings)  # 30 trainings
    tuned.round(4).to_csv(OUT_PATH)  # save the per-year results
    print("Saved", OUT_PATH.relative_to(ROOT))  # confirmation

    day10 = pd.read_csv(DAY10_PATH, index_col="fold")  # Day 10 results (Day 9 settings), same years
    summary = pd.DataFrame({"day9_settings": mean_std(day10), "tuned": mean_std(tuned)}).T  # one row per model
    print("\nSpace + time, mean ± std over 6 years:")  # header
    print(summary.to_string())  # the headline comparison
    wins = int((tuned["recall_top5"] > day10["recall_top5"]).sum())  # years where tuned recall is higher
    print(f"Tuned wins recall@top5% in {wins}/6 years")  # the honest per-year count

    mlflow.set_tracking_uri(TRACKING_URI)  # same mlflow.db as Days 8–11
    mlflow.set_experiment("wildfire-validation")  # next to the Day 10 scheme runs
    with mlflow.start_run(run_name="validation_spacetime_tuned"):  # one run for this check
        mlflow.set_tags({"stage": "validation", "scheme": "spacetime", "params": "tuned (configs/best_params.json)"})  # labels
        mlflow.log_metrics({f"mean_{m}": float(tuned[m].mean()) for m in METRICS})  # means, same names as Day 10
        mlflow.log_metrics({f"std_{m}": float(tuned[m].std()) for m in METRICS})  # spreads, same names as Day 10
        mlflow.log_artifact(str(OUT_PATH))  # the per-year table
    print("Logged to MLflow (wildfire-validation)")  # confirmation