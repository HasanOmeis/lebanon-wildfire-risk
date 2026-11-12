"""Download the API's runtime files (model + serving data) from the Hugging Face Hub when HF_REPO is set (Day 17)."""  # file description

import logging  # progress messages in the API's log
import os  # read HF_REPO from the environment
from pathlib import Path  # file paths that work on Windows and Linux

from huggingface_hub import HfApi, snapshot_download  # repo info (newest commit) and the folder downloader

ROOT = Path(__file__).resolve().parents[2]  # project root (src/api/hub_sync.py -> two folders up), same as predictor.py and store.py
SERVING_FILES = [  # exactly the files the API reads at startup; src/publish_hub.py uploads the same list
    "models/lgbm_tuned.txt",  # the trained LightGBM model
    "models/model_meta.json",  # model file name and feature list
    "models/risk_levels.json",  # cutoffs for low / moderate / high / extreme
    "data/serving/recent_features.parquet",  # last 400 days x 488 cells of features
    "data/serving/grid.geojson",  # the 488 cell polygons for the map
]  # end of the list

log = logging.getLogger("wildfire.api.hub")  # child of the API's logger, so its lines are easy to spot


def sync_from_hub() -> str | None:  # downloads the files if HF_REPO is set; returns the commit ID, or None when switched off
    repo = os.getenv("HF_REPO", "").strip()  # e.g. hasan-omeis/lebanon-fire-radar-serving; empty = local mode
    if not repo:  # laptop / Compose / tests: files come from the mounted folders instead
        return None  # nothing to do

    sha = HfApi().repo_info(repo, repo_type="dataset").sha  # the newest commit ID, so all files come from one upload
    log.info("Syncing %d files from %s @ %s ...", len(SERVING_FILES), repo, sha[:7])  # start line (short commit ID)
    snapshot_download(  # download only the listed files, skipping ones already up to date
        repo_id=repo,  # the dataset repo
        repo_type="dataset",  # it's a dataset repo, not a model repo
        revision=sha,  # that exact commit, even if a new upload lands mid-download
        allow_patterns=SERVING_FILES,  # only these 5 files (not README or anything else in the repo)
        local_dir=ROOT,  # keep the repo's folder layout under the project root -> models/..., data/serving/...
    )  # end of the download
    missing = [f for f in SERVING_FILES if not (ROOT / f).exists()]  # double-check every file arrived
    if missing:  # a file was renamed or deleted in the repo
        raise FileNotFoundError(f"Not found in {repo} @ {sha[:7]}: {missing}")  # stop: the API can't serve without them
    log.info("Synced %s @ %s", repo, sha[:7])  # done line
    return sha  # the commit now on disk