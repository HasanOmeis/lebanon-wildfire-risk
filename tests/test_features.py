import json  # read the saved feature list
import sys  # lets us tell Python where src/ is
from pathlib import Path  # clean file paths

import numpy as np  # small neighbor matrix for the synthetic test
import pandas as pd  # tables
import pytest  # test runner tools (fixtures, skip)

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))  # add the project's src/ folder to the import path
import features  # noqa: E402  # our feature code (imported after the path fix)

FEATURES_FILE = Path("data/processed/features.parquet")  # the real table built in Step 8


def make_daily(fire_days):  # turn a list of (cell, date) into the table build_panel expects
    return pd.DataFrame({  # one row per fire day
        "cell_id": [c for c, _ in fire_days],  # which cell burned
        "date": pd.to_datetime([d for _, d in fire_days]),  # on which day
        "n_detections": 1,  # one detection each
        "frp_sum": 10.0,  # fake fire power
    })  # end of the table


def build_synthetic(fire_days):  # run the time-based feature steps on 2 fake neighboring cells
    end = pd.Timestamp("2012-03-31")  # short panel: about 2 months
    A = np.array([[0, 1], [1, 0]], dtype="float32")  # cell 1 and cell 2 are neighbors
    panel = features.build_panel(make_daily(fire_days), [1, 2], end)  # base table with the label
    panel = features.add_fire_history(panel)  # rolling fire counts, days since fire, fire rate
    return features.add_neighbor_fires(panel, A, [1, 2])  # neighbor counts


def test_no_future_leakage():  # adding fires AFTER the cutoff must not change any row ON or BEFORE it
    cutoff = pd.Timestamp("2012-03-01")  # the "today" we check
    past = [(1, "2012-02-10"), (2, "2012-02-12")]  # fires before the cutoff
    future = past + [(1, "2012-03-20"), (2, "2012-03-21")]  # same, plus fires after the cutoff
    a = build_synthetic(past)  # features without the future fires
    b = build_synthetic(future)  # features with the future fires
    a_before = a[a["date"] <= cutoff].reset_index(drop=True)  # rows up to the cutoff, first version
    b_before = b[b["date"] <= cutoff].reset_index(drop=True)  # rows up to the cutoff, second version
    pd.testing.assert_frame_equal(a_before, b_before)  # must be identical, or a feature peeks into the future
    assert not a.equals(b)  # sanity: the future fires did change the later rows, so the test really bites


def test_label_is_next_day_fire_synthetic():  # fire_tomorrow on day t must equal fire_today on day t+1
    p = build_synthetic([(1, "2012-02-10")])  # one fire in cell 1
    row = p[(p["cell_id"] == 1) & (p["date"] == "2012-02-09")]  # the day before the fire
    assert row["fire_tomorrow"].item() == 1  # the label sees tomorrow's fire
    assert row["fire_today"].item() == 0  # but today itself had no fire


@pytest.fixture(scope="module")  # load the real table once for all tests below
def real():  # the saved features table
    if not FEATURES_FILE.exists():  # nothing built yet
        pytest.skip("run python src/features.py first")  # skip instead of failing
    return pd.read_parquet(FEATURES_FILE)  # about 2.6M rows


def test_unique_rows(real):  # each cell appears once per day
    assert not real.duplicated(["cell_id", "date"]).any()  # no duplicate cell-days


def test_complete_grid(real):  # every cell has every day
    assert len(real) == real["cell_id"].nunique() * real["date"].nunique()  # rows = cells x days


def test_label_matches_next_day(real):  # the real label equals the same cell's next-day fire
    nxt = real.sort_values(["cell_id", "date"]).groupby("cell_id")["fire_today"].shift(-1)  # recompute tomorrow's fire
    lab = real.sort_values(["cell_id", "date"])["fire_tomorrow"]  # the saved label, in the same order
    assert ((nxt == lab) | (nxt.isna() & lab.isna())).all()  # equal everywhere (both missing on each cell's last day)


def test_ndvi_never_from_future(real):  # NDVI must already be published on the day it's used
    assert (real["ndvi_age_days"].dropna() >= 0).all()  # a negative age would mean future vegetation data


def test_feature_list_matches_file(real):  # the JSON list and the table agree exactly
    cols = json.loads(Path("data/processed/feature_columns.json").read_text())  # saved feature names
    assert cols == [c for c in real.columns if c not in features.ID_COLS + [features.TARGET]]  # same names, same order