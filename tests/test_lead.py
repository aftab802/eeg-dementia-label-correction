import numpy as np
import pytest

pytest.importorskip("imblearn")
from sklearn.feature_selection import SelectKBest  # noqa: E402
from sklearn.model_selection import StratifiedKFold  # noqa: E402

from cybex_eeg.dementia import lead  # noqa: E402


def test_biomarkers():
    sf = 128.0
    x = np.vstack([np.sin(2 * np.pi * 10 * np.arange(512) / sf),
                   np.random.default_rng(0).normal(size=512)])
    b = lead.biomarkers(x, sf)
    assert b.shape == (2, 3)
    assert np.isclose(b[0, 1], b[0, 0] / 512)
    assert b[0, 2] < b[1, 2]  # pure tone has lower spectral entropy than noise


def _toy(n=120, d=12, seed=0):
    rng = np.random.default_rng(seed)
    y = np.r_[np.ones(80, int), np.zeros(n - 80, int)]
    X = rng.normal(size=(n, d)); X[y == 1, :3] += 1.5
    return X, y


def test_pipeline_learns_and_reports():
    X, y = _toy()
    r = lead.evaluate(X, y, n_splits=10)
    assert r["accuracy_pooled"] > 0.8 and len(r["confusion_matrix"]) == 2


def test_selection_and_smote_fitted_inside_folds(monkeypatch):
    X, y = _toy()
    seen = []
    orig = SelectKBest.fit

    def spy(self, Xf, yf):
        seen.append(len(yf))
        return orig(self, Xf, yf)
    monkeypatch.setattr(SelectKBest, "fit", spy)
    lead.evaluate(X, y, n_splits=10)
    folds = [len(tr) for tr, _ in StratifiedKFold(10, shuffle=True, random_state=42).split(X, y)]
    assert sorted(seen) == sorted(folds)  # selector never sees a held-out sample (and runs before SMOTE)


def test_too_few_features_is_an_error():
    X, y = _toy(d=3)
    with pytest.raises(ValueError):
        lead.evaluate(X, y, k=25, degree=2)
    assert lead.n_expanded(3, 4) == 34  # 3 channel-averaged biomarkers need degree >= 4 for k = 25


def test_grouped_cv_keeps_participants_together():
    X, y = _toy()
    groups = np.repeat(np.arange(60), 2)  # two recordings per participant, same label
    y = np.repeat(y[::2], 2)
    r = lead.evaluate(X, y, n_splits=10, groups=groups)
    assert r["grouped"] and r["participants_shared_across_folds"] == 0
