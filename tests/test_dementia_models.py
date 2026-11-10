import numpy as np
import pytest

pytest.importorskip("xgboost")
from cybex_eeg.dementia.models import subject_vote_groupkfold, xgb_gridsearch  # noqa: E402


def test_gridsearch_reports_best_and_nested():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(60, 6)); y = np.repeat([0, 1, 2], 20); X[:, 0] += y
    r = xgb_gridsearch(X, y, grid={"n_estimators": [20], "max_depth": [2, 3]}, n_jobs=1)
    assert 0 <= r["best_score"] <= 1 and "nested_cv_mean" in r and r["best_params"]["n_estimators"] == 20


def test_subject_vote_one_prediction_per_subject():
    rng = np.random.default_rng(0)
    g = np.repeat(np.arange(15), 10); y = np.repeat(np.arange(15) % 3, 10)
    X = rng.normal(size=(150, 50)); X[:, 0] += y
    r = subject_vote_groupkfold(X, y, g, k_best=10)
    assert len(r["subjects"]) == 15 and len(set(r["subjects"])) == 15
