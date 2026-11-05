import sys  # reads the scheme name typed after the script name
import time  # measures how long each fold takes

import geopandas as gpd  # reads the grid file (cell shapes and positions)
import lightgbm as lgb  # the gradient-boosted trees library
import mlflow  # experiment tracking
import numpy as np  # math on arrays
import pandas as pd  # tables
from sklearn.metrics import brier_score_loss  # checks if the probabilities themselves are honest

from train_baseline import (  # reuse Day 8 code so every scheme is judged the same way
    ROOT,  # project folder
    TARGET,  # label column: fire_tomorrow
    TRACKING_URI,  # MLflow database location
    load_data,  # loads features.parquet + the feature list
    evaluate,  # lift, PR-AUC, ROC-AUC, recall@top5%
)  # end of Day 8 imports
from train_lgbm import (  # reuse Day 9 code so the model is identical in every fold
    CATEGORICAL,  # month and day_of_week
    to_lgb_frame,  # turns a table into LightGBM-ready inputs
    build_params,  # the Day 9 LightGBM settings
)  # end of Day 9 imports

GRID_PATH = ROOT / "data" / "processed" / "lebanon_grid.gpkg"  # the 488 grid cells from Day 3
EXPERIMENT = "wildfire-validation"  # new MLflow experiment, kept apart from Day 8–9 runs
NUM_ROUNDS = 500  # number of trees: replace with the best iteration from your last Day 9 run
TEST_START = "2024-01-01"  # final test period from Day 8: no validation fold may touch it
TIME_TEST_YEARS = [2018, 2019, 2020, 2021, 2022, 2023]  # each time fold tests on one of these years
BLOCK_M = 25_000  # spatial block size in meters (25 km = about 5 x 5 grid cells)
N_SPATIAL_FOLDS = 5  # number of spatial folds


def fit_predict(train, test, feature_cols):  # train on one part, return fire scores for the other
    X_train, y_train = to_lgb_frame(train, feature_cols), train[TARGET]  # training inputs and labels
    n_pos = int(y_train.sum())  # fires in this training part
    pos_weight = (len(y_train) - n_pos) / n_pos  # imbalance ratio for this training part
    params = build_params(pos_weight)  # same settings as Day 9
    dtrain = lgb.Dataset(X_train, y_train, categorical_feature=CATEGORICAL)  # LightGBM's data format
    model = lgb.train(params, dtrain, num_boost_round=NUM_ROUNDS)  # fixed tree count, no peeking at the test part
    return model.predict(to_lgb_frame(test, feature_cols))  # fire scores for the test part


def precision_at_top(dates, y, score, top=0.05):  # share of flagged cells that really burn
    d = pd.DataFrame({"date": dates.values, "y": y.values, "score": np.asarray(score)})  # one small table: day, label, score
    rank = d.groupby("date")["score"].rank(ascending=False, method="first", pct=True)  # each cell's rank within its day (0 to 1)
    return d.loc[rank <= top, "y"].mean()  # fire rate among the riskiest 5% of cells each day


def score(test, proba):  # all of today's metrics for one set of predictions
    y = test[TARGET]  # true labels
    base = y.mean()  # fire rate in this test part
    brier = brier_score_loss(y, proba)  # mean squared error of the probabilities (lower is better)
    return {  # Day 8 metrics plus three new ones
        **evaluate(test, proba),  # lift, pr_auc, roc_auc, recall_top5
        "precision_top5": float(precision_at_top(test["date"], y, proba)),  # how many alerts were real fires
        "brier": brier,  # probability error
        "brier_skill": 1 - brier / (base * (1 - base)),  # > 0 beats always guessing the fire rate, < 0 is worse
    }  # end of metrics


def fit_and_score(train, test, feature_cols):  # train, predict and score in one call
    return score(test, fit_predict(train, test, feature_cols))  # all metrics for this fold


def leak_demo(df, feature_cols):  # Step 3: random split vs time split, to show the leak
    rng = np.random.default_rng(42)  # random generator with a fixed seed, so the split is repeatable
    is_test = rng.random(len(df)) < 0.2  # True for a random 20% of rows
    random_scores = fit_and_score(df[~is_test], df[is_test], feature_cols)  # random split: rows shuffled across all years
    train = df[df["date"] < "2022-01-01"]  # everything before 2022
    test = df[df["date"] >= "2022-01-01"]  # 2022–23 (2024+ was already removed), the Day 9 validation period
    time_scores = fit_and_score(train, test, feature_cols)  # honest split: past -> future
    return pd.DataFrame([random_scores, time_scores], index=["random_split", "time_split"])  # one row per split


def add_blocks(df):  # give every row the id of the 25 km block its cell sits in
    grid = gpd.read_file(GRID_PATH)  # cell shapes in meters (EPSG:32636)
    centers = grid.geometry.centroid  # the middle point of each cell
    bx = (centers.x // BLOCK_M).astype(int).astype(str)  # block column number (east-west)
    by = (centers.y // BLOCK_M).astype(int).astype(str)  # block row number (north-south)
    block_of_cell = pd.Series((bx + "_" + by).values, index=grid["cell_id"])  # lookup: cell_id -> block id like "28_371"
    return df.assign(block=df["cell_id"].map(block_of_cell))  # new column on every row


def block_fold_map(df):  # randomly spread the blocks over the spatial folds
    blocks = np.sort(df["block"].unique())  # every block id once, in a fixed order
    rng = np.random.default_rng(42)  # fixed seed, so every run makes the same folds
    rng.shuffle(blocks)  # mix the order so neighboring blocks land in different folds
    return {b: i % N_SPATIAL_FOLDS for i, b in enumerate(blocks)}  # block id -> fold number 0..4


def time_folds(df):  # walk-forward folds: train on the past, test on the next year
    for year in TIME_TEST_YEARS:  # one fold per test year
        test = df[df["date"].dt.year == year]  # the whole test year
        train = df[df["date"] < f"{year - 1}-12-31"]  # all earlier days, minus Dec 31 whose label is Jan 1 (the gap)
        yield f"time_{year}", train, test  # fold name + its two parts


def spatial_folds(df):  # hold out whole regions, all years together
    fold_of_row = df["block"].map(block_fold_map(df))  # which fold each row's block belongs to
    for k in range(N_SPATIAL_FOLDS):  # one fold per group of blocks
        test = df[fold_of_row == k]  # the held-out regions
        train = df[fold_of_row != k]  # every other region
        yield f"space_{k}", train, test  # fold name + its two parts


def record_fold(rows, name, test, scores, seconds):  # save one fold: in the table, on screen and in MLflow
    rows.append({"fold": name, "test_fires": int(test[TARGET].sum()), **scores})  # add a row to the results table
    print(f"{name:15} fires={rows[-1]['test_fires']:>4}  pr_auc={scores['pr_auc']:.4f}  "  # one progress line per fold
          f"recall_top5={scores['recall_top5']:.4f}  ({seconds:.0f} s)")  # second half of the line
    with mlflow.start_run(run_name=name, nested=True):  # child run inside the scheme's parent run
        mlflow.log_param("test_fires", rows[-1]["test_fires"])  # how many fires this fold was judged on
        mlflow.log_metrics(scores)  # every metric of this fold


def run_scheme(folds, feature_cols):  # train and score every fold of the time or spatial scheme
    rows = []  # one dict of results per fold
    for name, train, test in folds:  # loop over the folds
        start = time.time()  # start the timer
        scores = fit_and_score(train, test, feature_cols)  # train on this fold's train part, score its test part
        record_fold(rows, name, test, scores, time.time() - start)  # save the fold everywhere
    return pd.DataFrame(rows).set_index("fold")  # all folds as one table


def run_spacetime(df, feature_cols):  # strict scheme: future year AND unseen regions
    fold_of_row = df["block"].map(block_fold_map(df))  # same block -> fold assignment as the spatial scheme
    rows = []  # one dict of results per test year
    for year in TIME_TEST_YEARS:  # one result per test year
        start = time.time()  # start the timer for this year
        is_year = df["date"].dt.year == year  # rows in the test year
        is_past = df["date"] < f"{year - 1}-12-31"  # rows before the test year, with the 1-day gap
        test = df[is_year]  # the full test year, all 488 cells
        proba = pd.Series(np.nan, index=test.index)  # empty score slot for every test row, filled region by region
        for k in range(N_SPATIAL_FOLDS):  # one model per held-out region group
            train = df[is_past & (fold_of_row != k)]  # past years, other regions only
            part = test[fold_of_row[is_year] == k]  # this year's rows in the held-out regions
            proba.loc[part.index] = fit_predict(train, part, feature_cols)  # fill in their scores
        scores = score(test, proba.values)  # score the whole year: every cell scored by a model blind to it
        record_fold(rows, f"spacetime_{year}", test, scores, time.time() - start)  # save the year everywhere
    return pd.DataFrame(rows).set_index("fold")  # all years as one table


def run_and_log(scheme, df, feature_cols):  # run one scheme inside its own MLflow parent run
    with mlflow.start_run(run_name=f"validation_{scheme}"):  # parent run that holds the folds
        mlflow.set_tags({"stage": "validation", "scheme": scheme})  # searchable labels
        mlflow.log_params({"num_rounds": NUM_ROUNDS, "block_m": BLOCK_M,  # settings shared by every fold
                           "n_spatial_folds": N_SPATIAL_FOLDS, "test_years": str(TIME_TEST_YEARS)})  # more settings
        if scheme == "spacetime":  # the strict scheme has its own loop
            results = run_spacetime(df, feature_cols)  # 6 years x 5 regions = 30 trainings
        else:  # time or spatial
            results = run_scheme(SCHEMES[scheme](df), feature_cols)  # train and score every fold
        summary = results.drop(columns="test_fires").agg(["mean", "std"])  # average and spread across folds
        mlflow.log_metrics({f"mean_{k}": v for k, v in summary.loc["mean"].items()})  # e.g. mean_pr_auc
        mlflow.log_metrics({f"std_{k}": v for k, v in summary.loc["std"].items()})  # e.g. std_pr_auc
        out_path = ROOT / "reports" / f"validation_{scheme}.csv"  # one results file per scheme
        results.round(4).to_csv(out_path)  # save it for the comparison in Step 8
        mlflow.log_artifact(str(out_path))  # also store the table inside the parent run
    print(summary.round(4))  # the score to quote is mean ± std
    print("Saved", out_path.relative_to(ROOT))  # confirmation


SCHEMES = {"time": time_folds, "spatial": spatial_folds}  # scheme name -> function that makes its folds


if __name__ == "__main__":  # runs only when you call this file directly
    scheme = sys.argv[1] if len(sys.argv) > 1 else "time"  # e.g. "python src/validate.py spatial"; default is time
    df, feature_cols = load_data()  # same table and features as Day 9
    df = df[df["date"] < TEST_START]  # drop 2024+ right away, so no fold can ever see the test period
    df = add_blocks(df)  # attach the 25 km block id to every row
    print(f"Rows: {len(df):,}  Features: {len(feature_cols)}  Blocks: {df['block'].nunique()}")  # quick size check

    if scheme == "leak":  # the Step 3 demo, kept for the README
        print(leak_demo(df, feature_cols).round(4))  # random vs time split
        sys.exit()  # nothing more to do for the demo

    mlflow.set_tracking_uri(TRACKING_URI)  # same mlflow.db as Day 8–9
    mlflow.set_experiment(EXPERIMENT)  # create wildfire-validation the first time, reuse it after
    to_run = ["time", "spatial", "spacetime"] if scheme == "all" else [scheme]  # "all" runs the three schemes in a row
    for name in to_run:  # one scheme at a time
        print(f"\n=== {name} ===")  # header so the output is easy to read
        run_and_log(name, df, feature_cols)  # run, print, save and log this scheme