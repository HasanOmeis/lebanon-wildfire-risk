import json  # reads and writes the settings file

import optuna  # opens the saved study

from tune import ROOT, STORAGE, STUDY_NAME, suggest_params  # same study and the exact search-space code

BEST_PATH = ROOT / "configs" / "best_params.json"  # the one file every later script reads


def save_best_params():  # write the winning trial's full settings to configs/best_params.json
    study = optuna.load_study(study_name=STUDY_NAME, storage=STORAGE)  # open the saved search
    best = study.best_trial  # highest mean PR-AUC
    fixed = optuna.trial.FixedTrial(best.params)  # fake trial that always answers with the winner's values
    params, num_rounds, weight_power = suggest_params(fixed)  # rebuild the settings exactly as the search did
    out = {  # everything needed to retrain the winner
        "source": f"optuna study '{STUDY_NAME}', trial {best.number} of {len(study.trials)}",  # where it came from
        "cv_scheme": "walk-forward time folds, test years 2018-2023",  # how it was scored
        "cv_mean_pr_auc": best.value,  # its tuning score
        "cv_mean_recall_top5": best.user_attrs["mean_recall_top5"],  # its operational score
        "num_rounds": num_rounds,  # number of trees
        "weight_power": weight_power,  # fire weight = (non-fires / fires) ** this
        "lgb_params": params,  # the LightGBM settings dict
    }  # end of settings
    BEST_PATH.parent.mkdir(exist_ok=True)  # create the configs folder if it doesn't exist yet
    BEST_PATH.write_text(json.dumps(out, indent=2))  # save with readable indentation
    return out  # handy for printing


def load_best_params():  # read the saved winner (used by later scripts)
    best = json.loads(BEST_PATH.read_text())  # turn the file back into a dict
    return best["lgb_params"], best["num_rounds"], best["weight_power"]  # same three values as suggest_params


if __name__ == "__main__":  # runs only when you call this file directly
    optuna.logging.set_verbosity(optuna.logging.WARNING)  # hide the "loading study" info lines
    out = save_best_params()  # write the file
    print(json.dumps(out, indent=2))  # show what was saved
    print("Saved", BEST_PATH.relative_to(ROOT))  # confirmation
    params, num_rounds, weight_power = load_best_params()  # read it back, to test the loader
    print(f"Loader check: {len(params)} settings, {num_rounds} trees, weight_power={weight_power:.4f}")  # should match above