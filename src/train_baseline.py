import json  # reads the saved list of feature names
from pathlib import Path  # builds file paths that work on Windows

import pandas as pd  # loads and handles the feature table

import numpy as np  # random numbers for the sanity test
from sklearn.metrics import average_precision_score, roc_auc_score  # ranking metrics for rare events
import time  # measures how long training takes
from sklearn.impute import SimpleImputer  # fills missing values
from sklearn.linear_model import LogisticRegression  # the baseline model
from sklearn.pipeline import make_pipeline  # chains the steps into one object
from sklearn.preprocessing import StandardScaler  # puts features on the same scale

import matplotlib  # plotting library
matplotlib.use("Agg")  # draw charts to files only, no pop-up windows
import matplotlib.pyplot as plt  # creates the figure for the PR curve
import mlflow  # experiment tracking
import mlflow.sklearn  # saves scikit-learn models to MLflow
from mlflow.models import infer_signature  # records the model's input/output format
from sklearn.metrics import PrecisionRecallDisplay  # draws the precision-recall curve

ROOT = Path(__file__).resolve().parents[1]  # project folder (one level above src/)
FEATURES_PATH = ROOT / "data" / "processed" / "features.parquet"  # Day 7 model-ready table
COLUMNS_PATH = ROOT / "data" / "processed" / "feature_columns.json"  # Day 7 list of the 48 features
TARGET = "fire_tomorrow"  # label: 1 if the cell burns on day t+1, else 0
VAL_START = "2022-01-01"  # first day of the validation period
TEST_START = "2024-01-01"  # first day of the test period (untouched until final scoring)
NAIVE_BASELINES = {  # rule-based scores: name -> feature column used directly as the risk score
    "climatology": "cell_fire_rate",  # cells that burned often in the past
    "persistence": "fires_last_7d",  # cells that burned in the last 7 days
}
TRACKING_URI = f"sqlite:///{(ROOT / 'mlflow.db').as_posix()}"  # MLflow database file in the project folder
EXPERIMENT = "wildfire-baseline"  # groups all of today's runs together



def load_data():  # load the table and keep only rows that have a label
    df = pd.read_parquet(FEATURES_PATH)  # read the full feature table
    df["date"] = pd.to_datetime(df["date"])  # make sure date is a real datetime
    df = df.dropna(subset=[TARGET])  # drop rows with no tomorrow (the last day in the data)
    df[TARGET] = df[TARGET].astype(int)  # store the label as 0/1 integers
    with open(COLUMNS_PATH) as f:  # open the feature list file
        feature_cols = json.load(f)  # list of feature column names
    return df, feature_cols  # give back the table and the feature names

def split_by_time(df):  # cut the table into train / validation / test by date
    train = df[df["date"] < VAL_START]  # everything before 2022
    val = df[(df["date"] >= VAL_START) & (df["date"] < TEST_START)]  # 2022 and 2023
    test = df[df["date"] >= TEST_START]  # 2024 to the latest day
    return train, val, test  # give back the three parts

def recall_at_top(dates, y, score, top=0.05):  # share of fires caught when we flag the top 5% of cells each day
    d = pd.DataFrame({"date": dates.values, "y": y.values, "score": np.asarray(score)})  # one small table: day, label, score
    rank = d.groupby("date")["score"].rank(ascending=False, method="first", pct=True)  # each cell's rank within its day (0 to 1)
    flagged = rank <= top  # True for the riskiest 5% of cells on each day
    return d.loc[flagged, "y"].sum() / d["y"].sum()  # fires inside flagged cells divided by all fires


def evaluate(part, score):  # all metrics for one set of predictions on one data split
    y = part[TARGET]  # true labels for this split
    pr_auc = average_precision_score(y, score)  # main metric: area under the precision-recall curve
    return {  # results as a dict, ready to log to MLflow later
        "lift": float(pr_auc / y.mean()),  # how many times better than random (plain float for clean printing)
        "pr_auc": pr_auc,  # higher is better, random ≈ fire rate
        "roc_auc": roc_auc_score(y, score),  # 0.5 = random, 1.0 = perfect ranking
        "recall_top5": float(recall_at_top(part["date"], y, score)),  # share of fires inside each day's top-5% cells
    }


def build_logreg(C=1.0):  # create the imputer -> scaler -> logistic regression pipeline
    return make_pipeline(  # each step's output feeds the next step
        SimpleImputer(strategy="median", add_indicator=True),  # fill gaps with the train median + add "was missing" flags
        StandardScaler(),  # mean 0, std 1 for every column
        LogisticRegression(C=C, class_weight="balanced", max_iter=1000),  # C = regularization strength (smaller = simpler model)
    )  # end of pipeline


def pr_curve_figure(y, score, name):  # precision-recall chart for one model on validation
    fig, ax = plt.subplots(figsize=(6, 4))  # empty figure, 6x4 inches
    PrecisionRecallDisplay.from_predictions(y, score, name=name, ax=ax)  # draw the curve from labels + scores
    ax.axhline(y.mean(), linestyle="--", color="grey", label="random")  # the random-guess line (= fire rate)
    ax.legend()  # show the labels
    return fig  # hand back the figure to log it


def log_run(name, params, metrics, y, score, model=None, X_example=None):  # save one run to MLflow
    with mlflow.start_run(run_name=name):  # open a new run; it closes automatically at the end of the block
        mlflow.set_tags({"stage": "baseline", "val_period": f"{VAL_START} to {TEST_START}"})  # searchable labels
        mlflow.log_params({**params, "train_end": VAL_START})  # settings of this run + where training stopped
        mlflow.log_metrics({f"val_{k}": v for k, v in metrics.items()})  # scores, prefixed "val_" (test comes later)
        fig = pr_curve_figure(y, score, name)  # build the PR chart
        mlflow.log_figure(fig, "pr_curve_val.png")  # store the chart inside the run
        plt.close(fig)  # free the memory used by the figure
        if model is not None:  # only real models get saved (rules have nothing to save)
            X_example = X_example.astype("float64")  # floats can hold missing values, integers can't
            signature = infer_signature(X_example, model.predict_proba(X_example)[:, 1])  # input columns + output type
            mlflow.sklearn.log_model(  # save the fitted pipeline inside the run
                model, name="model", signature=signature, input_example=X_example,  # model + its input/output format
                skops_trusted_types=["numpy.dtype"],  # allow the imputer's stored column types (safe: no code inside)
            )  # end of log_model

if __name__ == "__main__":  # runs only when you call this file directly
    df, feature_cols = load_data()  # load everything
    mlflow.set_tracking_uri(TRACKING_URI)  # tell MLflow where to save runs
    mlflow.set_experiment(EXPERIMENT)  # create the experiment the first time, reuse it after
    print("Rows:", f"{len(df):,}")  # total labelled cell-days
    print("Features:", len(feature_cols))  # should be 48
    print("Dates:", df["date"].min().date(), "->", df["date"].max().date())  # time range covered
    print("Positives:", f"{df[TARGET].sum():,}")  # number of fire-tomorrow rows
    print("Positive rate:", f"{df[TARGET].mean():.4%}")  # share of rows that are fires
    yearly = df.groupby(df["date"].dt.year)[TARGET].agg(["sum", "mean"])  # fires and rate per year
    yearly.columns = ["fires", "rate"]  # clearer column names
    print(yearly)  # shows which years have enough fires to validate and test on
    missing = df[feature_cols].isna().mean().sort_values(ascending=False)  # share missing per feature
    print("Most-missing features:")  # header for the next print
    print(missing.head(5).round(3))  # top 5, mainly weather until the download finishes
    train, val, test = split_by_time(df)  # make the three time periods
    for name, part in [("train", train), ("val", val), ("test", test)]:  # loop over each part
        print(f"{name:5} rows={len(part):>9,}  fires={part[TARGET].sum():>5,}  rate={part[TARGET].mean():.4%}  "  # size, fires and rate
              f"{part['date'].min().date()} -> {part['date'].max().date()}")  # date range of this part
    assert len(train) + len(val) + len(test) == len(df)  # no row lost or counted twice
    results = []  # one dict of metrics per model, collected for the summary table
    for name, col in NAIVE_BASELINES.items():  # score each rule on the validation period
        scores = evaluate(val, val[col])  # the feature value itself is the risk score
        print(f"{name:12}", {k: round(v, 4) for k, v in scores.items()})  # one line of metrics per baseline
        log_run(name, {"model": "rule", "score_column": col}, scores, val[TARGET], val[col])  # save the rule's run
        results.append({"model": name, **scores})  # remember this rule's validation scores
    X_train, y_train = train[feature_cols], train[TARGET]  # training inputs and labels
    X_val = val[feature_cols]  # validation inputs
    model = build_logreg()  # a fresh, untrained pipeline
    start = time.time()  # start the timer
    model.fit(X_train, y_train)  # learn from 2012–2021
    print(f"Trained in {time.time() - start:.0f} s")  # training time
    val_proba = model.predict_proba(X_val)[:, 1]  # predicted fire probability for each validation row
    scores = evaluate(val, val_proba)  # same metrics as the baselines
    print(f"{'logreg':12}", {k: round(v, 4) for k, v in scores.items()})  # compare with the two lines above
    log_run(  # save the logistic regression run
        "logreg",  # run name shown in the MLflow UI
        {"model": "logistic_regression", "C": 1.0, "imputer": "median+indicator",  # model settings
         "class_weight": "balanced", "n_features": len(feature_cols)},  # more settings
        scores, val[TARGET], val_proba,  # metrics, labels and predictions for the chart
        model=model, X_example=X_val.head(5),  # the fitted pipeline + 5 example rows
    )  # end of log_run
    results.append({"model": "logreg", **scores})  # add the logistic regression's validation scores
    table = pd.DataFrame(results).set_index("model").round(4)  # one row per model, one column per metric
    out_path = ROOT / "reports" / "baseline_results.csv"  # where the table is saved
    out_path.parent.mkdir(parents=True, exist_ok=True)  # create reports/ if it doesn't exist
    table.to_csv(out_path)  # save the table for the README and later comparison with LightGBM
    print(table)  # show the final comparison

