import json  # reads the Day 10 validation config
import sys  # reads the target number of trials typed after the script name
import time  # measures how long loading and trials take

import lightgbm as lgb  # the gradient-boosted trees library
import mlflow  # experiment tracking
import numpy as np  # averages across folds
import optuna  # the hyperparameter search library
import pandas as pd  # tables

from train_baseline import (  # reuse Day 8 code so tuning is judged like every other model
    ROOT,  # project folder
    TARGET,  # label column: fire_tomorrow
    TRACKING_URI,  # MLflow database location
    load_data,  # loads features.parquet + the feature list
)  # end of Day 8 imports
from train_lgbm import (  # reuse Day 9 code so the inputs are identical
    CATEGORICAL,  # month and day_of_week
    to_lgb_frame,  # turns a table into LightGBM-ready inputs
)  # end of Day 9 imports
from validate import time_folds, score  # reuse Day 10's walk-forward folds and its full metric set

CONFIG_PATH = ROOT / "reports" / "validation_config.json"  # the official setup locked on Day 10
CONFIG = json.loads(CONFIG_PATH.read_text())  # turn the JSON file into a Python dict
TEST_START = CONFIG["final_test_start"]  # 2024-01-01: the locked final test period
STORAGE = f"sqlite:///{(ROOT / 'optuna.db').as_posix()}"  # study database file in the project folder
STUDY_NAME = "lgbm-time-folds"  # name of the study inside that file
DONE_STATES = (optuna.trial.TrialState.COMPLETE, optuna.trial.TrialState.PRUNED)  # trials that count as finished
EXPERIMENT = "wildfire-tuning"  # new MLflow experiment for today's trials

DAY9_PARAMS = {  # Day 9's hand-picked settings, written in the search-space names
    "learning_rate": 0.02,  # Day 9 step size
    "num_leaves": 15,  # Day 9 tree size
    "min_child_samples": 1000,  # Day 9 rows per leaf
    "feature_fraction": 0.5,  # Day 9 share of features per tree
    "bagging_fraction": 0.8,  # Day 9 share of rows per tree
    "lambda_l1": 1e-3,  # Day 9 had no L1 penalty; 0.001 is the smallest value in the range (practically zero)
    "lambda_l2": 10.0,  # Day 9 L2 penalty
    "min_gain_to_split": 0.0,  # Day 9 used LightGBM's default (0)
    "num_rounds": 500,  # Day 10's fixed tree count
    "weight_power": 0.5,  # square root of the imbalance ratio, as on Day 9
}  # end of Day 9 settings


def prepare_folds(df, feature_cols):  # build each fold's LightGBM inputs once, reused by every trial
    folds = []  # one dict per fold
    for name, train, test in time_folds(df):  # Day 10's 6 walk-forward folds
        dtrain = lgb.Dataset(  # LightGBM's training data format
            to_lgb_frame(train, feature_cols),  # training inputs
            train[TARGET],  # training labels
            categorical_feature=CATEGORICAL,  # month and day_of_week as categories
            params={"feature_pre_filter": False},  # keep every feature, since Optuna changes min_child_samples between trials
            free_raw_data=False,  # keep the data so the Dataset can be reused by many trials
        )  # end of Dataset
        folds.append({  # everything one fold needs
            "name": name,  # e.g. "time_2018"
            "dtrain": dtrain,  # training data, ready for lgb.train
            "X_test": to_lgb_frame(test, feature_cols),  # test inputs
            "test": test[["date", "cell_id", TARGET]],  # only the columns scoring needs (saves memory)
            "n_pos": int(train[TARGET].sum()),  # fires in this training part
            "n_neg": int((train[TARGET] == 0).sum()),  # non-fire rows in this training part
        })  # end of fold dict
    return folds  # list of 6 ready folds


def suggest_params(trial):  # Optuna picks one value for each setting in this trial
    params = {  # settings dict passed to lgb.train
        "objective": "binary",  # predict fire (1) vs no fire (0)
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),  # step size; log scale tries small values as often as big ones
        "num_leaves": trial.suggest_int("num_leaves", 7, 63, log=True),  # tree complexity (Day 9 used 15)
        "min_child_samples": trial.suggest_int("min_child_samples", 100, 5000, log=True),  # rows needed per leaf (Day 9 used 1,000)
        "feature_fraction": trial.suggest_float("feature_fraction", 0.3, 1.0),  # share of features each tree sees (Day 9 used 0.5)
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1.0),  # share of rows each tree sees; 1.0 = bagging off
        "bagging_freq": 1,  # redraw the rows for every tree (no effect when bagging_fraction is 1.0)
        "lambda_l1": trial.suggest_float("lambda_l1", 1e-3, 10.0, log=True),  # penalty that can zero out weak leaves
        "lambda_l2": trial.suggest_float("lambda_l2", 1e-3, 100.0, log=True),  # penalty on extreme leaf values (Day 9 used 10)
        "min_gain_to_split": trial.suggest_float("min_gain_to_split", 0.0, 1.0),  # a split must improve the loss at least this much
        "seed": 42,  # same random choices every run, so results are repeatable
        "verbose": -1,  # hide LightGBM's internal warnings
    }  # end of settings
    num_rounds = trial.suggest_int("num_rounds", 100, 1000, step=50)  # number of trees, tuned instead of early stopping
    weight_power = trial.suggest_float("weight_power", 0.0, 1.0)  # fire weight = ratio ** power (0 = no weighting, 0.5 = Day 9's sqrt)
    return params, num_rounds, weight_power  # everything a trial needs to train


def make_objective(folds):  # build the function Optuna calls once per trial
    def objective(trial):  # one trial = one set of settings tested on all 6 folds
        params, num_rounds, weight_power = suggest_params(trial)  # this trial's settings
        fold_scores = []  # metrics of each fold
        for step, f in enumerate(folds):  # train and score every time fold; step = 0, 1, 2 ... 5
            fold_params = {**params, "scale_pos_weight": (f["n_neg"] / f["n_pos"]) ** weight_power}  # weight from this fold's own imbalance
            model = lgb.train(fold_params, f["dtrain"], num_boost_round=num_rounds)  # fixed tree count, no peeking at the test year
            proba = model.predict(f["X_test"])  # fire scores for the test year
            s = score(f["test"], proba)  # Day 10's full metric set for this fold
            fold_scores.append(s)  # keep it
            trial.set_user_attr(f"{f['name']}_pr_auc", s["pr_auc"])  # store this fold's PR-AUC on the trial
            trial.set_user_attr(f"{f['name']}_recall_top5", s["recall_top5"])  # store this fold's recall on the trial
            running = float(np.mean([x["pr_auc"] for x in fold_scores]))  # mean PR-AUC over the folds done so far
            trial.report(running, step)  # tell Optuna how this trial is doing after this fold
            if trial.should_prune():  # below the median of other trials at the same fold?
                raise optuna.TrialPruned()  # stop this trial early and move on
        mean = {k: float(np.mean([s[k] for s in fold_scores])) for k in fold_scores[0]}  # average of every metric over the folds
        trial.set_user_attr("mean_recall_top5", mean["recall_top5"])  # operational number, kept next to the main score
        trial.set_user_attr("mean_roc_auc", mean["roc_auc"])  # overall ranking quality
        return mean["pr_auc"]  # the number Optuna tries to maximize
    return objective  # hand the function to Optuna


def make_study():  # open the saved study, or create it the first time
    sampler = optuna.samplers.TPESampler(seed=42)  # smart search that learns from past trials; fixed seed = repeatable
    pruner = optuna.pruners.MedianPruner(n_startup_trials=8, n_warmup_steps=2)  # first 8 trials run fully; never prune before 3 folds
    return optuna.create_study(  # create or reopen the study
        study_name=STUDY_NAME,  # which study inside the file
        storage=STORAGE,  # saved in optuna.db, so it survives crashes and restarts
        load_if_exists=True,  # reopen it on later runs instead of failing
        direction="maximize",  # higher mean PR-AUC is better
        sampler=sampler,  # how new settings are chosen
        pruner=pruner,  # when bad trials are stopped early
    )  # end of create_study


def count_done(study):  # how many trials are finished (completed or pruned)
    return len(study.get_trials(deepcopy=False, states=DONE_STATES))  # crashed or interrupted trials don't count


def logged_trial_numbers():  # which trials are already in MLflow
    runs = mlflow.search_runs(experiment_names=[EXPERIMENT], output_format="list")  # every run of the tuning experiment
    return {int(r.data.tags["trial_number"]) for r in runs if "trial_number" in r.data.tags}  # their trial numbers as a set


def log_trial(trial):  # save one finished trial as one MLflow run
    with mlflow.start_run(run_name=f"trial_{trial.number:03d}"):  # e.g. "trial_007", so runs sort in order
        mlflow.set_tags({  # searchable labels
            "study": STUDY_NAME,  # which study this run belongs to
            "trial_number": str(trial.number),  # used by the sync to skip trials already logged
            "state": trial.state.name,  # COMPLETE or PRUNED
        })  # end of tags
        mlflow.log_params(trial.params)  # the settings Optuna chose
        mlflow.log_metrics(trial.user_attrs)  # per-fold scores (and means, if the trial finished)
        mlflow.log_metric("folds_done", len(trial.intermediate_values))  # 6 if complete, fewer if pruned
        if trial.value is not None:  # only complete trials have a final score
            mlflow.log_metric("mean_pr_auc", trial.value)  # the number Optuna maximized


def sync_mlflow(study, _trial=None):  # log every finished trial that MLflow doesn't have yet
    logged = logged_trial_numbers()  # trials already saved
    new = 0  # how many we log now
    for t in study.get_trials(deepcopy=False, states=DONE_STATES):  # every finished trial
        if t.number not in logged:  # missing from MLflow?
            log_trial(t)  # save it
            new += 1  # count it
    return new  # used for the status line (Optuna ignores it when called as a callback)


if __name__ == "__main__":  # runs only when you call this file directly
    target = int(sys.argv[1]) if len(sys.argv) > 1 else 1  # e.g. "python src/tune.py 50"; default is 1 trial
    mlflow.set_tracking_uri(TRACKING_URI)  # same mlflow.db as Days 8–10
    mlflow.set_experiment(EXPERIMENT)  # create wildfire-tuning the first time, reuse it after
    study = make_study()  # open or create the saved study
    if len(study.trials) == 0:  # brand-new study?
        study.enqueue_trial(DAY9_PARAMS)  # first trial = Day 9 settings, the baseline to beat
    print(f"Logged to MLflow now: {sync_mlflow(study)} earlier trial(s)")  # catch up on trials finished before a crash
    remaining = target - count_done(study)  # trials still needed to reach the target
    print(f"Finished trials: {count_done(study)}  Target: {target}  To run: {max(remaining, 0)}")  # status line
    if remaining > 0:  # only load data if there's work to do
        start = time.time()  # start the timer
        df, feature_cols = load_data()  # same table and features as Days 8–10
        df = df[df["date"] < TEST_START]  # drop 2024+ right away: tuning must never see the final test
        folds = prepare_folds(df, feature_cols)  # build the 6 folds once
        print(f"Folds ready in {time.time() - start:.0f} s")  # loading time
        study.optimize(  # run the search
            make_objective(folds),  # the function that trains and scores one trial
            n_trials=remaining,  # only the missing trials
            callbacks=[sync_mlflow],  # after each trial, save it to MLflow
        )  # end of optimize
    best = study.best_trial  # highest mean PR-AUC so far
    print(f"Best: trial {best.number}  mean PR-AUC={best.value:.4f}  "  # main score of the best trial
          f"mean recall@top5%={best.user_attrs['mean_recall_top5']:.4f}")  # its operational score