"""Upload the API's runtime files (model + serving data) to the Hugging Face dataset repo in one commit (Day 17)."""  # file description

import datetime as dt  # today's date for the commit message
import logging  # progress messages, same style as the Day 15 pipeline
import os  # read HF_TOKEN and HF_REPO from the environment
from pathlib import Path  # file paths that work on Windows and Linux

from dotenv import load_dotenv  # reads .env into environment variables
from huggingface_hub import CommitOperationAdd, HfApi  # one "add this file" operation per file, and the Hub client

from src.api.hub_sync import SERVING_FILES  # the one shared list: the API downloads exactly what we upload

ROOT = Path(__file__).resolve().parents[1]  # project root (src/publish_hub.py -> one folder up)

log = logging.getLogger("pipeline.hub")  # same naming as the pipeline's loggers, so the daily log reads cleanly


def publish(message: str | None = None) -> str:  # uploads all files as one commit; returns the commit ID
    load_dotenv(ROOT / ".env")  # load HF_TOKEN and HF_REPO from the project's .env
    token = os.environ.get("HF_TOKEN")  # write token (only on the laptop, never in the cloud API)
    repo = os.environ.get("HF_REPO")  # e.g. hasan-omeis/lebanon-fire-radar-serving
    if not token or not repo:  # one of them is missing or empty
        raise RuntimeError("HF_TOKEN and HF_REPO must both be set in .env")  # stop with a clear message

    missing = [f for f in SERVING_FILES if not (ROOT / f).exists()]  # files that aren't on disk
    if missing:  # never upload half a set
        raise FileNotFoundError(f"Missing files, nothing uploaded: {missing}")  # name them

    total_mb = sum((ROOT / f).stat().st_size for f in SERVING_FILES) / 1e6  # total upload size in MB, for the log
    ops = [CommitOperationAdd(path_in_repo=f, path_or_fileobj=str(ROOT / f)) for f in SERVING_FILES]  # same path in the repo as in the project
    today = dt.datetime.now(dt.timezone.utc).date()  # UTC date, same rule as the pipeline
    message = message or f"Serving update {today}"  # default commit message

    log.info("Uploading %d files (%.1f MB) to %s ...", len(SERVING_FILES), total_mb, repo)  # start line
    info = HfApi(token=token).create_commit(  # one commit with every file in it
        repo_id=repo,  # target repo
        repo_type="dataset",  # it's a dataset repo, not a model repo
        operations=ops,  # the file uploads
        commit_message=message,  # shows in the repo's history on the website
    )  # end of the commit
    log.info("Uploaded: %s", info.commit_url)  # link to this commit on the website
    return info.oid  # the commit ID (useful in pipeline logs)


if __name__ == "__main__":  # run alone: python -m src.publish_hub
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(name)s | %(message)s")  # same log format as the pipeline
    publish()  # manual upload with the default dated message