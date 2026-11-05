import geopandas as gpd  # reads the grid (cell latitudes)
import mlflow  # experiment tracking
import optuna  # FixedTrial rebuilds the Day 9 settings
import pandas as pd  # tables
from sklearn.metrics import average_precision_score  # PR-AUC for each group

from train_baseline import ROOT, TARGET, TRACKING_URI, load_data  # project folder, label, MLflow location, data loader
from validate import GRID_PATH, TEST_START, score  # grid file, 2024-01-01 test start, Day 10 metric set
from tune import DAY9_PARAMS, suggest_params  # Day 9 settings + the exact search-space code
from best_params import load_best_params  # the tuned winner
from confirm_tuned import fit_predict_tuned  # same training function as the space+time check

TRAIN_END = "2023-12-31"  # train on days before this: drops Dec 31, 2023, whose label is Jan 1, 2024 (1-day gap)
SOUTH_LAT = 33.35  # cells south of this latitude = the south (same 64 cells as the Day 6 EDA)
EVENTS_START = "2024-01-01"  # security-event period inside the test set starts with the test set
EVENTS_END = "2024-11-27"  # ...and ends with the Nov 27, 2024 ceasefire
PRED_PATH = ROOT / "data" / "processed" / "test_predictions.parquet"  # saved scores for Days 12 and 16
OVERALL_PATH = ROOT / "reports" / "final_test.csv"  # overall test scores per model
BREAKDOWN_PATH = ROOT / "reports" / "final_test_breakdown.csv"  # scores per group per model
SHOW = ["pr_auc", "lift", "roc_auc", "recall_top5", "precision_top5", "brier_skill"]  # overall columns to print


def day9_settings():  # rebuild the Day 9 settings in the same format as the tuned ones
    return suggest_params(optuna.trial.FixedTrial(DAY9_PARAMS))  # params, num_rounds, weight_power


def flag_top(test, proba, top=0.05):  # True for cells in Lebanon's top 5% riskiest on their day
    rank = proba.groupby(test["date"]).rank(ascending=False, method="first", pct=True)  # each cell's national rank within its day
    return rank <= top  # flagged cells


def group_scores(test, proba, flagged, mask):  # scores for one group of rows (e.g. the south)
    y = test[TARGET]  # true labels
    fires = mask & (y == 1)  # this group's fires
    return {  # this group's numbers
        "fires": int(fires.sum()),  # how many fires the group has
        "pr_auc": average_precision_score(y[mask], proba[mask]),  # ranking quality inside the group
        "recall_top5": float(flagged[fires].mean()),  # share of the group's fires that were flagged nationally
        "precision_top5": float(y[mask & flagged].mean()),  # share of the group's flagged cells that burned
    }  # end of group scores


if __name__ == "__main__":  # runs only when you call this file directly
    df, feature_cols = load_data()  # same table and features as Days 8–11
    train = df[df["date"] < TRAIN_END]  # 2012–2023, with the 1-day gap
    test = df[df["date"] >= TEST_START]  # 2024+: unlocked now, once
    print(f"Train: {train['date'].min():%Y-%m-%d} to {train['date'].max():%Y-%m-%d}  fires={int(train[TARGET].sum()):,}")  # train range
    print(f"Test:  {test['date'].min():%Y-%m-%d} to {test['date'].max():%Y-%m-%d}  fires={int(test[TARGET].sum()):,}")  # test range

    grid = gpd.read_file(GRID_PATH)  # the 488 cells with their lat column
    south = test["cell_id"].isin(grid.loc[grid["lat"] < SOUTH_LAT, "cell_id"])  # True for rows in southern cells
    events = test["date"].between(EVENTS_START, EVENTS_END)  # True for rows in the 2024 security-event period
    year = test["date"].dt.year  # test year of every row
    groups = {  # name -> which rows belong to the group
        "all": pd.Series(True, index=test.index),  # every test row
        "south": south,  # southern cells
        "rest_of_lebanon": ~south,  # every other cell
        "security_events_2024": events,  # Jan 1 – Nov 27, 2024
        "outside_events": ~events,  # the rest of the test period
        "year_2024": year == 2024,  # each test year on its own
        "year_2025": year == 2025,  # NDVI forward-filled from late June to mid-November
        "year_2026": year == 2026,  # partial year (up to late October)
    }  # end of groups

    models = {"day9": day9_settings(), "tuned": load_best_params()}  # the two models, scored once each
    preds = test[["date", "cell_id", TARGET]].copy()  # table for the saved scores
    overall, rows = {}, []  # overall scores per model; one row per model per group
    for name, settings in models.items():  # train and score each model
        proba = pd.Series(fit_predict_tuned(train, test, feature_cols, settings), index=test.index)  # test scores
        preds[f"proba_{name}"] = proba  # keep them
        overall[name] = score(test, proba.values)  # Day 10 metric set on the whole test period
        flagged = flag_top(test, proba)  # national top 5% each day
        for g, mask in groups.items():  # every group
            rows.append({"model": name, "group": g, **group_scores(test, proba, flagged, mask)})  # its scores
        print(f"Scored {name}")  # progress

    overall = pd.DataFrame(overall).T  # one row per model
    breakdown = pd.DataFrame(rows)  # one row per model per group
    overall.round(4).to_csv(OVERALL_PATH)  # save overall scores
    breakdown.round(4).to_csv(BREAKDOWN_PATH, index=False)  # save the breakdown
    preds.to_parquet(PRED_PATH, index=False)  # save the scores for Days 12 and 16
    print("Saved", OVERALL_PATH.relative_to(ROOT), BREAKDOWN_PATH.relative_to(ROOT), PRED_PATH.relative_to(ROOT))  # confirmation

    pd.set_option("display.width", 200)  # wide enough for one line per row
    print("\nFinal test (2024+), overall:")  # header
    print(overall[SHOW].round(4).to_string())  # overall comparison
    wide = breakdown.pivot(index="group", columns="model", values=["fires", "recall_top5", "pr_auc"])  # groups as rows, models side by side
    print("\nBreakdown (recall = share of the group's fires inside the national top 5% each day):")  # header
    print(wide.reindex(list(groups)).round(4).to_string())  # in the same order as the groups dict

    mlflow.set_tracking_uri(TRACKING_URI)  # same mlflow.db as Days 8–11
    mlflow.set_experiment("wildfire-final-test")  # its own experiment: this happens once
    with mlflow.start_run(run_name="final_test_2024plus"):  # one run for the whole evaluation
        mlflow.set_tags({"stage": "final_test", "train": "2012-2023", "test": "2024+"})  # labels
        for name in overall.index:  # each model
            mlflow.log_metrics({f"{name}_{m}": float(overall.loc[name, m]) for m in SHOW})  # e.g. tuned_recall_top5
        mlflow.log_artifact(str(OVERALL_PATH))  # overall table
        mlflow.log_artifact(str(BREAKDOWN_PATH))  # breakdown table
    print("Logged to MLflow (wildfire-final-test)")  # confirmation