import matplotlib  # plotting library
matplotlib.use("Agg")  # draw charts to files only, no pop-up windows
import optuna  # opens the saved study
import pandas as pd  # tables

from tune import ROOT, STORAGE, STUDY_NAME  # same study file and name as the search
from validate import TIME_TEST_YEARS  # the 6 test years (2018–2023)

FIG_DIR = ROOT / "reports" / "figures"  # where the charts go (same folder as the EDA charts)
TOP_PATH = ROOT / "reports" / "tuning_top_trials.csv"  # where the top-10 table goes
COMPARE_PATH = ROOT / "reports" / "tuning_vs_day9.csv"  # where the year-by-year comparison goes
BASELINE_TRIAL = 0  # trial 0 = the Day 9 settings
SHOW = [  # columns worth reading in the top-10 table
    "number",  # trial number
    "value",  # mean PR-AUC (what Optuna maximized)
    "user_attrs_mean_recall_top5",  # mean recall@top5%
    "params_weight_power",  # fire weighting (0 = none, 0.5 = Day 9)
    "params_min_child_samples",  # rows per leaf
    "params_num_leaves",  # tree size
    "params_learning_rate",  # step size
    "params_num_rounds",  # number of trees
]  # end of column list


def save_chart(ax, name):  # save one Optuna chart as a PNG
    fig = ax.figure  # the figure that holds the chart
    fig.set_size_inches(8, 5)  # readable size
    fig.savefig(FIG_DIR / name, dpi=120, bbox_inches="tight")  # save without cutting off labels
    print("Saved", (FIG_DIR / name).relative_to(ROOT))  # confirmation with a short path


def compare_years(base, best):  # per-year scores of the Day 9 trial vs the best trial
    rows = []  # one row per test year
    for year in TIME_TEST_YEARS:  # 2018 ... 2023
        rows.append({  # this year's four scores
            "year": year,  # test year
            "day9_pr_auc": base.user_attrs[f"time_{year}_pr_auc"],  # Day 9 PR-AUC this year
            "tuned_pr_auc": best.user_attrs[f"time_{year}_pr_auc"],  # tuned PR-AUC this year
            "day9_recall_top5": base.user_attrs[f"time_{year}_recall_top5"],  # Day 9 recall@top5% this year
            "tuned_recall_top5": best.user_attrs[f"time_{year}_recall_top5"],  # tuned recall@top5% this year
        })  # end of row
    table = pd.DataFrame(rows).set_index("year")  # one row per year
    table["pr_auc_gain_%"] = 100 * (table["tuned_pr_auc"] / table["day9_pr_auc"] - 1)  # relative PR-AUC change
    table["recall_gain_pts"] = 100 * (table["tuned_recall_top5"] - table["day9_recall_top5"])  # recall change in % points
    return table  # the comparison table


if __name__ == "__main__":  # runs only when you call this file directly
    optuna.logging.set_verbosity(optuna.logging.WARNING)  # hide the "loading study" info lines
    study = optuna.load_study(study_name=STUDY_NAME, storage=STORAGE)  # open the saved search
    df = study.trials_dataframe()  # one row per trial with its settings, scores and state
    print(df["state"].value_counts().to_string())  # how many trials completed vs pruned

    ax = optuna.visualization.matplotlib.plot_optimization_history(study)  # score of every trial + best-so-far line
    save_chart(ax, "11_optuna_history.png")  # save it
    ax = optuna.visualization.matplotlib.plot_param_importances(study)  # which settings changed the score most
    save_chart(ax, "11_param_importance.png")  # save it

    done = df[df["state"] == "COMPLETE"]  # only trials that ran all 6 folds have a final score
    top = done.sort_values("value", ascending=False).head(10)  # the 10 best trials
    top.to_csv(TOP_PATH, index=False)  # save every column for the README
    print("Saved", TOP_PATH.relative_to(ROOT))  # confirmation
    pd.set_option("display.width", 200)  # wide enough to show the table on one line per trial
    print(top[SHOW].round(4).to_string(index=False))  # print the key columns of the top 10

    base = study.trials[BASELINE_TRIAL]  # the Day 9 trial
    best = study.best_trial  # the tuned winner
    table = compare_years(base, best)  # year-by-year comparison
    table.round(4).to_csv(COMPARE_PATH)  # save it for the README
    print(f"\nDay 9 (trial {base.number}) vs tuned (trial {best.number}), per test year:")  # header
    print(table.round(4).to_string())  # print the table
    pr_wins = int((table["tuned_pr_auc"] > table["day9_pr_auc"]).sum())  # years where tuned PR-AUC is higher
    rec_wins = int((table["tuned_recall_top5"] > table["day9_recall_top5"]).sum())  # years where tuned recall is higher
    print(f"Tuned wins PR-AUC in {pr_wins}/6 years, recall@top5% in {rec_wins}/6 years")  # the honest summary