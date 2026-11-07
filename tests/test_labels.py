"""Every model feature must have a plain-English label for the map."""  # file description

from pathlib import Path  # file paths

import pytest  # skip support

from src.api.predictor import FEATURE_LABELS, MODEL_DIR, Predictor  # the labels and the model loader

pytestmark = pytest.mark.skipif(  # skip on a fresh clone where the model isn't built yet
    not Path(MODEL_DIR / "lgbm_tuned.txt").exists(),  # the model file is missing...
    reason="model not built: run src/train_final.py first",  # ...so say why
)  # end of the skip rule


def test_every_feature_has_a_label():  # no code names should reach users
    features = Predictor().feature_cols  # the 52 features the model really uses
    unlabeled = [f for f in features if f not in FEATURE_LABELS]  # any without a label
    assert unlabeled == [], f"add labels for: {unlabeled}"  # fail with the exact names to add


def test_no_stale_labels():  # no labels left over for features the model no longer uses
    features = set(Predictor().feature_cols)  # the model's features
    stale = [f for f in FEATURE_LABELS if f not in features]  # labels with no matching feature
    assert stale == [], f"remove labels for: {stale}"  # fail with the exact names to remove