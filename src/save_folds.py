import json  # writes the settings file

from train_baseline import ROOT, load_data  # project folder + the feature table loader
from validate import (  # reuse today's fold logic so the saved folds match exactly
    TEST_START,  # 2024-01-01: the locked final test start
    TIME_TEST_YEARS,  # 2018–2023 test years
    BLOCK_M,  # 25 km block size
    N_SPATIAL_FOLDS,  # 5 spatial folds
    NUM_ROUNDS,  # tree count used today
    add_blocks,  # cell -> 25 km block
    block_fold_map,  # block -> spatial fold
)  # end of imports from validate


if __name__ == "__main__":  # runs only when you call this file directly
    df, _ = load_data()  # the feature table (feature list not needed here)
    df = df[df["date"] < TEST_START]  # same rows as validate.py
    cells = add_blocks(df[["cell_id"]].drop_duplicates())  # one row per cell with its block id
    cells["spatial_fold"] = cells["block"].map(block_fold_map(df.pipe(add_blocks)))  # the block's fold number 0..4
    out_cells = ROOT / "data" / "processed" / "cell_folds.parquet"  # where the cell -> fold table goes
    cells.sort_values("cell_id").to_parquet(out_cells, index=False)  # save it
    print(cells["spatial_fold"].value_counts().sort_index())  # cells per spatial fold, should be roughly even

    config = {  # the official validation setup, in one small file
        "tuning_scheme": "time",  # Optuna (Day 11) scores trials on the walk-forward time folds
        "report_scheme": "spacetime",  # the headline number comes from the strict scheme
        "time_test_years": TIME_TEST_YEARS,  # one test year per time fold
        "gap_days": 1,  # Dec 31 dropped from training: its label is the first test day
        "final_test_start": TEST_START,  # 2024+ is never used for tuning
        "block_m": BLOCK_M,  # spatial block size in meters
        "n_spatial_folds": N_SPATIAL_FOLDS,  # number of spatial folds
        "num_rounds": NUM_ROUNDS,  # tree count used in today's validation
        "cell_folds_file": "data/processed/cell_folds.parquet",  # where the cell -> fold table lives
    }  # end of settings
    out_config = ROOT / "reports" / "validation_config.json"  # small text file, safe to commit
    out_config.write_text(json.dumps(config, indent=2))  # save it with readable indentation
    print("Saved", out_cells.relative_to(ROOT), "and", out_config.relative_to(ROOT))  # confirmation