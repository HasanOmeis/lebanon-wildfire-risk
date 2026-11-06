from pathlib import Path   # handles file paths the same way on Windows and Linux

import matplotlib          # plotting library that SHAP draws with
matplotlib.use("Agg")      # draw to files only, no pop-up windows
import matplotlib.pyplot as plt   # used to save and close each figure
import mlflow              # logs the plots and table to the experiment tracker
import numpy as np         # math for the base value and probabilities
import pandas as pd        # reads the saved sample and SHAP values
import shap                # SHAP plotting functions

from explain import load_model, prepare_X   # reuse the model loader and the category fix from explain.py

ROOT = Path(__file__).resolve().parents[1]                              # project root folder (one level above src)
SAMPLE_PATH = ROOT / "data" / "processed" / "shap_sample.parquet"       # explained rows saved in step 4
SHAP_PATH = ROOT / "data" / "processed" / "shap_values.parquet"         # SHAP values saved in step 4
FIG_DIR = ROOT / "reports" / "figures"                                  # folder for the plot images
IMPORTANCE_PATH = ROOT / "reports" / "shap_importance.csv"              # table of mean |SHAP| per feature
TRACKING_URI = f"sqlite:///{(ROOT / 'mlflow.db').as_posix()}"           # same MLflow database used since Day 8
RUN_ID = "9a26a41ca85a466091a3548883ddd82e"                             # the lgbm_tuned_all_data run that saved lgbm_tuned.txt
TARGET = "fire_tomorrow"                                                # label column: 1 if the cell burns the next day
TOP_N = 15                                                              # how many features each plot shows
DEPENDENCE_FEATURES = ["days_since_fire", "elev_mean", "ndvi", "vpd_max", "tmax_mean_7d"]   # features to plot one by one


def to_prob(log_odds: np.ndarray) -> np.ndarray:                        # function that turns log-odds into a probability
    return 1 / (1 + np.exp(-log_odds))                                  # the sigmoid function


def load_explanation() -> tuple[shap.Explanation, pd.DataFrame, np.ndarray]:   # rebuilds the SHAP object and returns the rows and raw scores
    shap_df = pd.read_parquet(SHAP_PATH)                                # SHAP values, one column per feature
    sample = pd.read_parquet(SAMPLE_PATH)                               # the matching rows (ids, date, target, features)
    model = load_model()                                                # load the tuned model
    raw = model.predict(prepare_X(sample, model), raw_score=True)       # each row's raw log-odds score
    base = float(np.mean(raw - shap_df.sum(axis=1).to_numpy()))         # base value = score minus all pushes (same for every row)
    data = sample[shap_df.columns].astype(float)                        # the 52 features as plain numbers, so plots can color by value
    expl = shap.Explanation(                                            # bundle everything SHAP plots need
        values=shap_df.to_numpy(),                                      # the SHAP pushes
        base_values=np.full(len(sample), base),                         # the starting point, one per row
        data=data.to_numpy(),                                           # the feature values
        feature_names=list(shap_df.columns),                            # the feature names
    )                                                                   # end of the Explanation
    return expl, sample, raw                                            # return the explanation, the rows and the raw scores


def save_current(name: str) -> None:                                    # function that saves the current figure
    plt.savefig(FIG_DIR / name, dpi=150, bbox_inches="tight")          # write the image at good quality, trimmed edges
    plt.close("all")                                                    # close it so the next plot starts clean
    print("Saved", name)                                                # confirm the file was written


def importance_table(expl: shap.Explanation) -> pd.DataFrame:           # function that ranks features by average impact
    mean_abs = abs(expl.values).mean(axis=0)                            # average size of each feature's push, ignoring direction
    table = pd.DataFrame({"feature": expl.feature_names, "mean_abs_shap": mean_abs})   # one row per feature
    return table.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)  # biggest impact first


def dependence_plots(expl: shap.Explanation) -> None:                   # function that saves one dependence plot per feature
    for name in DEPENDENCE_FEATURES:                                    # go through the chosen features
        shap.plots.scatter(expl[:, name], show=False)                   # x = feature value, y = its SHAP push, one dot per row
        plt.axhline(0, color="grey", linewidth=0.8)                     # line at zero: above it raises risk, below it lowers risk
        save_current(f"shap_dep_{name}.png")                            # save the plot with the feature name in the file name


def waterfall_plots(expl: shap.Explanation, sample: pd.DataFrame, raw: np.ndarray) -> None:   # one waterfall for a risky row, one for a safe row
    fire_idx = np.where(sample[TARGET] == 1)[0]                         # positions of rows where a fire really happened next day
    calm_idx = np.where(sample[TARGET] == 0)[0]                         # positions of rows with no fire next day
    high = fire_idx[np.argmax(raw[fire_idx])]                           # the fire row the model scored highest
    low = calm_idx[np.argmin(raw[calm_idx])]                            # the no-fire row the model scored lowest
    for label, i in [("high", high), ("low", low)]:                     # draw both cases the same way
        row = sample.iloc[i]                                            # that row's ids, date and features
        prob = float(to_prob(raw[i]))                                   # turn its score into a probability
        print(f"{label}: cell {row['cell_id']}, date {pd.to_datetime(row['date']).date()}, risk {prob:.2%}")   # show which cell and day it is
        shap.plots.waterfall(expl[i], max_display=12, show=False)       # the step-by-step push from base value to final score
        save_current(f"shap_waterfall_{label}.png")                     # save the waterfall image


def log_to_mlflow(table: pd.DataFrame, n_rows: int) -> None:            # function that attaches the SHAP results to the tuned run
    mlflow.set_tracking_uri(TRACKING_URI)                               # point MLflow at the project's database
    with mlflow.start_run(run_id=RUN_ID):                               # reopen the existing tuned run instead of making a new one
        for png in sorted(FIG_DIR.glob("shap_*.png")):                  # only the SHAP images, not the EDA figures
            mlflow.log_artifact(str(png), artifact_path="shap")         # upload each image into a "shap" folder of the run
        mlflow.log_artifact(str(IMPORTANCE_PATH), artifact_path="shap") # upload the importance table too
        top10 = table.head(10)                                          # the 10 most important features
        mlflow.log_metrics({f"shap_{f}": float(v) for f, v in zip(top10["feature"], top10["mean_abs_shap"])})   # their scores as metrics, for comparing runs later
        mlflow.set_tag("shap_sample_rows", n_rows)                      # record how many rows were explained
        mlflow.set_tag("shap_top_feature", table.loc[0, "feature"])     # record the single most important feature
    print("Logged SHAP results to run", RUN_ID)                         # confirm the upload


if __name__ == "__main__":                                              # run this part only when the file is run directly
    FIG_DIR.mkdir(parents=True, exist_ok=True)                          # create reports/figures if it doesn't exist
    expl, sample, raw = load_explanation()                              # load the SHAP values, the rows and their scores
    table = importance_table(expl)                                      # rank the features
    table.to_csv(IMPORTANCE_PATH, index=False)                          # save the ranking as a CSV
    print(table.head(TOP_N).to_string())                                # print the top features with their scores
    shap.plots.bar(expl, max_display=TOP_N, show=False)                 # bar chart of mean |SHAP| for the top features
    save_current("shap_bar.png")                                        # save the bar chart
    shap.plots.beeswarm(expl, max_display=TOP_N, show=False)            # one dot per row: position = push, color = feature value
    save_current("shap_beeswarm.png")                                   # save the beeswarm plot
    dependence_plots(expl)                                              # save the five dependence plots
    waterfall_plots(expl, sample, raw)                                  # save the two waterfall plots
    log_to_mlflow(table, len(sample))                                   # attach everything to the tuned MLflow run