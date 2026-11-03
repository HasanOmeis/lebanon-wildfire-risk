import time  # measures how long training takes

import lightgbm as lgb  # the gradient-boosted trees library
import pandas as pd  # tables

import matplotlib  # plotting library
matplotlib.use("Agg")  # draw charts to files only, no pop-up windows
import matplotlib.pyplot as plt  # creates the figures
import mlflow  # experiment tracking
import mlflow.lightgbm  # saves LightGBM models to MLflow


from train_baseline import (  # reuse Day 8 code so both models are judged identically
    ROOT,  # project folder
    TARGET,  # label column name: fire_tomorrow
    TRACKING_URI,  # MLflow database location
    EXPERIMENT,  # MLflow experiment name (wildfire-baseline)
    load_data,  # loads features.parquet + the feature list
    split_by_time,  # train <2022, val 2022–23, test 2024+
    evaluate,  # lift, PR-AUC, ROC-AUC, recall@top5%
    pr_curve_figure,  # precision-recall chart
)  # end of imports from train_baseline

CATEGORICAL = ["month", "day_of_week"]  # labels, not quantities: month 12 isn't "bigger" than month 1



def to_lgb_frame(part, feature_cols):  # turn one data split into LightGBM-ready inputs
    X = part[feature_cols].copy()  # copy so the original table isn't changed
    X["month"] = pd.Categorical(X["month"], categories=range(1, 13))  # months 1–12, same codes in every split
    X["day_of_week"] = pd.Categorical(X["day_of_week"], categories=range(7))  # 0 = Monday ... 6 = Sunday, same codes everywhere
    return X  # missing values stay as NaN on purpose: LightGBM handles them itself


def build_params(pos_weight):  # hand-picked starting settings (Optuna tunes these on Day 11)
    return {  # settings dict passed to lgb.train
        "objective": "binary",  # predict fire (1) vs no fire (0)
        "metric": "average_precision",  # PR-AUC on validation, the same main metric as Day 8
        "learning_rate": 0.02,  # smaller corrections: slower, steadier learning
        "num_leaves": 15,  # simpler trees, which can't carve out tiny groups of rows
        "min_child_samples": 1000,  # each leaf needs 1,000 rows, so no leaf is built around a handful of fires
        "feature_fraction": 0.5,  # each tree sees only half the features, so it can't lean on one drifting column
        "bagging_fraction": 0.8,  # each tree trains on a random 80% of the rows
        "bagging_freq": 1,  # redraw that 80% of rows for every tree        
        "lambda_l2": 10.0,  # stronger penalty on extreme leaf values
        "scale_pos_weight": pos_weight ** 0.5,  # square root of 529 ≈ 23: a moderate push toward fires
        "seed": 42,  # same random choices every run, so results are repeatable        
        "verbose": -1,  # hide LightGBM's internal warnings
    }  # end of settings

def importance_figure(model, top=20):  # bar chart of the features that helped the model most
    fig, ax = plt.subplots(figsize=(7, 6))  # empty figure, 7x6 inches
    lgb.plot_importance(model, ax=ax, importance_type="gain", max_num_features=top)  # gain = total improvement each feature brought
    ax.set_title(f"Top {top} features by gain")  # chart title
    fig.tight_layout()  # keep long feature names from being cut off
    return fig  # hand back the figure to log it


def log_lgbm_run(params, scores, model, y_val, val_proba):  # save the LightGBM run to MLflow
    with mlflow.start_run(run_name="lightgbm"):  # open a new run; it closes at the end of the block
        mlflow.set_tags({"stage": "lightgbm", "val_period": "2022-01-01 to 2024-01-01"})  # searchable labels
        mlflow.log_params({**params, "best_iteration": model.best_iteration})  # all settings + the tree count kept
        mlflow.log_metrics({f"val_{k}": v for k, v in scores.items()})  # scores with the same "val_" names as Day 8
        fig = pr_curve_figure(y_val, val_proba, "lightgbm")  # PR chart, same style as the baselines
        mlflow.log_figure(fig, "pr_curve_val.png")  # store it in the run
        plt.close(fig)  # free memory
        fig = importance_figure(model)  # feature importance chart
        mlflow.log_figure(fig, "feature_importance_gain.png")  # store it in the run
        plt.close(fig)  # free memory
        mlflow.lightgbm.log_model(model, name="model")  # save the trained trees inside the run



if __name__ == "__main__":  # runs only when you call this file directly
    df, feature_cols = load_data()  # same table and 52 features as the baseline
    train, val, test = split_by_time(df)  # same three time periods as the baseline
    print("Features:", len(feature_cols))  # should be 52
    print("Categorical present:", [c for c in CATEGORICAL if c in feature_cols])  # check both columns exist
    print(f"train={len(train):,}  val={len(val):,}  test={len(test):,}")  # same sizes as train_baseline printed
    
    X_train, y_train = to_lgb_frame(train, feature_cols), train[TARGET]  # training inputs and labels
    X_val, y_val = to_lgb_frame(val, feature_cols), val[TARGET]  # validation inputs and labels
    print(X_train[CATEGORICAL].dtypes)  # both should say "category"
    print("Missing values kept:", int(X_train.isna().sum().sum()))  # NaNs left for LightGBM to handle
   
   
    n_pos = int(y_train.sum())  # number of fire rows in training
    n_neg = len(y_train) - n_pos  # number of non-fire rows in training
    pos_weight = n_neg / n_pos  # how many times rarer fires are than non-fires
    print(f"Fires: {n_pos:,}  Non-fires: {n_neg:,}  scale_pos_weight = {pos_weight:.0f}")  # the imbalance ratio


    params = build_params(pos_weight)  # the settings for this run
    dtrain = lgb.Dataset(X_train, y_train, categorical_feature=CATEGORICAL)  # training data in LightGBM's format
    dval = lgb.Dataset(X_val, y_val, reference=dtrain)  # validation data, using the same bins as training
    start = time.time()  # start the timer
    model = lgb.train(  # train the boosted trees
        params,  # the settings above
        dtrain,  # learn from 2012–2021
        num_boost_round=2000,  # upper limit on the number of trees
        valid_sets=[dtrain, dval],  # score both training and validation after every tree
        valid_names=["train", "val"],  # names in the log; early stopping watches "val" only
        callbacks=[lgb.early_stopping(100), lgb.log_evaluation(100)],  # stop after 100 trees without gain; print every 100
    )  # end of training
    print(f"Trained in {time.time() - start:.0f} s, best iteration = {model.best_iteration}")  # time + best tree count

    val_proba = model.predict(X_val, num_iteration=model.best_iteration)  # fire scores for 2022–23 using the best tree count
    scores = evaluate(val, val_proba)  # lift, PR-AUC, ROC-AUC, recall@top5% (same function as Day 8)
    baseline = pd.read_csv(ROOT / "reports" / "baseline_results.csv", index_col="model")  # Day 8 scores, rerun today with full weather
    table = pd.concat([baseline, pd.DataFrame([scores], index=["lightgbm"])])  # add LightGBM as a new row
    print(table.round(4))  # all four models side by side

    mlflow.set_tracking_uri(TRACKING_URI)  # same mlflow.db as the baselines
    mlflow.set_experiment(EXPERIMENT)  # same wildfire-baseline experiment
    log_lgbm_run(params, scores, model, y_val, val_proba)  # save this run next to the Day 8 runs
    print("Logged LightGBM run to MLflow")  # confirmation

    out_path = ROOT / "reports" / "model_results.csv"  # baselines + LightGBM in one file
    table.round(4).to_csv(out_path, index_label="model")  # save it for the README and the Day 10 comparison
    print("Saved", out_path.relative_to(ROOT))  # confirmation with a short path