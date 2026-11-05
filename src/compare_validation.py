import matplotlib  # plotting library
matplotlib.use("Agg")  # draw charts to files only, no pop-up windows
import matplotlib.pyplot as plt  # creates the figure
import pandas as pd  # tables

from train_baseline import ROOT  # project folder

SCHEMES = ["time", "spatial", "spacetime"]  # the three validation schemes, from easiest to strictest
METRICS = ["lift", "pr_auc", "roc_auc", "recall_top5", "precision_top5", "brier_skill"]  # columns for the summary


if __name__ == "__main__":  # runs only when you call this file directly
    folds = {s: pd.read_csv(ROOT / "reports" / f"validation_{s}.csv", index_col="fold") for s in SCHEMES}  # load each scheme's fold table

    rows = {}  # one summary row per scheme
    for s, table in folds.items():  # loop over the schemes
        mean, std = table[METRICS].mean(), table[METRICS].std()  # average and spread across folds
        rows[s] = {m: f"{mean[m]:.4f} ± {std[m]:.4f}" for m in METRICS}  # text like "0.0152 ± 0.0058"
    summary = pd.DataFrame(rows).T  # schemes as rows, metrics as columns
    summary.index.name = "scheme"  # name the index column in the CSV
    print(summary.to_string())  # show the full table without cutting columns
    summary.to_csv(ROOT / "reports" / "validation_summary.csv")  # save it for the README

    fig, ax = plt.subplots(figsize=(7, 4))  # empty figure, 7x4 inches
    for i, s in enumerate(SCHEMES):  # one column of dots per scheme
        values = folds[s]["pr_auc"]  # PR-AUC of every fold in this scheme
        ax.scatter([i] * len(values), values, alpha=0.7, label=None)  # one dot per fold
        ax.hlines(values.mean(), i - 0.25, i + 0.25, color="black")  # short black line at the mean
    ax.set_xticks(range(len(SCHEMES)))  # one tick per scheme
    ax.set_xticklabels(SCHEMES)  # label the ticks with the scheme names
    ax.set_ylabel("PR-AUC per fold")  # y-axis label
    ax.set_title("Validation schemes: every fold, with the mean")  # chart title
    fig.tight_layout()  # keep labels from being cut off
    out_path = ROOT / "reports" / "validation_comparison.png"  # where the chart is saved
    fig.savefig(out_path, dpi=150)  # save a sharp image for the README
    print("Saved", out_path.relative_to(ROOT), "and reports/validation_summary.csv")  # confirmation