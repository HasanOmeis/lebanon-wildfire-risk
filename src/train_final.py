import json  # writes the model metadata file

import lightgbm as lgb  # the gradient-boosted trees library
import mlflow  # experiment tracking
import mlflow.lightgbm  # saves LightGBM models to MLflow
import numpy as np  # compares predictions
import pandas as pd  # tables

from train_baseline import ROOT, TARGET, TRACKING_URI, load_data  # project folder, label, MLflow location, data loader
from train_lgbm import CATEGORICAL, to_lgb_frame  # same LightGBM inputs as Days 9–11
from best_params import BEST_PATH, load_best_params  # the tuned winner and its file

MODEL_DIR = ROOT / "models"  # model folder (ignored by Git: models are rebuilt, not stored)
MODEL_PATH = MODEL_DIR / "lgbm_tuned.txt"  # LightGBM's own text format
META_PATH = MODEL_DIR / "model_meta.json"  # what the API needs to know about the model
TEST_PATH = ROOT / "reports" / "final_test.csv"  # the honest 2024+ scores from Step 14
REGISTERED_NAME = "lebanon-wildfire-lgbm"  # name in the MLflow Model Registry


def train_model(train, feature_cols, settings):  # train one model with the given settings
    params, num_rounds, weight_power = settings  # unpack the winner
    y = train[TARGET]  # labels
    n_pos = int(y.sum())  # fires
    weight = ((len(y) - n_pos) / n_pos) ** weight_power  # fire weight, same formula as the search
    dtrain = lgb.Dataset(to_lgb_frame(train, feature_cols), y, categorical_feature=CATEGORICAL)  # LightGBM data
    model = lgb.train({**params, "scale_pos_weight": weight}, dtrain, num_boost_round=num_rounds)  # fixed tree count
    return model, weight  # the trained trees and the weight used


if __name__ == "__main__":  # runs only when you call this file directly
    df, feature_cols = load_data()  # every labeled row, 2012 to the latest day
    settings = load_best_params()  # tuned settings from configs/best_params.json
    model, weight = train_model(df, feature_cols, settings)  # train on all the data
    print(f"Trained on {df['date'].min():%Y-%m-%d} to {df['date'].max():%Y-%m-%d}: "  # training range
          f"{len(df):,} rows, {int(df[TARGET].sum()):,} fires, {model.num_trees()} trees")  # size check

    MODEL_DIR.mkdir(exist_ok=True)  # make sure the models folder exists
    model.save_model(str(MODEL_PATH))  # save the trees as a text file
    test_scores = pd.read_csv(TEST_PATH, index_col=0).loc["tuned"].round(4).to_dict()  # the tuned row of the final test
    meta = {  # everything the API needs besides the trees
        "model_file": MODEL_PATH.name,  # which file holds the trees
        "feature_cols": feature_cols,  # exact input columns, in order
        "categorical": CATEGORICAL,  # columns that must be categories (month, day_of_week)
        "trained_from": f"{df['date'].min():%Y-%m-%d}",  # first training day
        "trained_to": f"{df['date'].max():%Y-%m-%d}",  # last training day
        "scale_pos_weight": weight,  # fire weight used (≈ 1)
        "params_file": str(BEST_PATH.relative_to(ROOT)),  # where the settings came from
        "test_2024plus_scores": test_scores,  # honest performance estimate (model trained on 2012–2023)
    }  # end of metadata
    META_PATH.write_text(json.dumps(meta, indent=2))  # save it
    print("Saved", MODEL_PATH.relative_to(ROOT), "and", META_PATH.relative_to(ROOT))  # confirmation

    sample = to_lgb_frame(df.tail(5000), feature_cols)  # last 5,000 rows, as test inputs
    reloaded = lgb.Booster(model_file=str(MODEL_PATH))  # load the saved file back, as the API will
    same = np.allclose(model.predict(sample), reloaded.predict(sample))  # do both give the same scores?
    print(f"Reload check: {'identical predictions' if same else 'MISMATCH'}")  # must say identical

    mlflow.set_tracking_uri(TRACKING_URI)  # same mlflow.db as Days 8–11
    mlflow.set_experiment("wildfire-production")  # experiment for the model that goes live
    with mlflow.start_run(run_name="lgbm_tuned_all_data"):  # one run for the production model
        mlflow.set_tags({"stage": "production", "trained_to": meta["trained_to"]})  # labels
        mlflow.log_params({**settings[0], "num_rounds": settings[1], "scale_pos_weight": weight})  # full settings
        mlflow.log_metrics({f"test_2024plus_{k}": v for k, v in test_scores.items()})  # honest test scores
        mlflow.log_artifact(str(META_PATH))  # metadata file
        mlflow.lightgbm.log_model(model, name="model", registered_model_name=REGISTERED_NAME)  # save + register a new version
    print(f"Registered in MLflow as '{REGISTERED_NAME}'")  # confirmation