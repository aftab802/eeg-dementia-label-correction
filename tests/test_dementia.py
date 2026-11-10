import numpy as np
import pandas as pd
import pytest

from cybex_eeg.dementia import ds004504 as D
from cybex_eeg.dementia.gsp import GSP_FEATURES, gsp_features
from cybex_eeg.dementia.qeeg import BANDS_NOTEBOOK, epoch_slowing_features, qeeg28


def _participants(n=88):
    groups = ["A"] * 36 + ["F"] * 23 + ["C"] * 29
    df = pd.DataFrame({"participant_id": [f"sub-{i:03d}" for i in range(1, n + 1)], "Group": groups[:n]})
    df["subject_id"] = df["participant_id"].map(D.canonical_id)
    df["diagnosis"] = df["Group"].map(D.GROUP_NAMES)
    return df


@pytest.mark.parametrize("raw,expected", [("sub-007", "sub-007"), ("sub-007_task-eyesclosed_eeg.set", "sub-007"),
                                          (7, "sub-007"), ("S7", "sub-007")])
def test_canonical_id(raw, expected):
    assert D.canonical_id(raw) == expected


def test_keyed_merge_survives_row_shuffle_positional_does_not():
    parts = _participants()
    feats = pd.DataFrame({"participant_id": parts["participant_id"], "f": np.arange(88.0)})
    shuffled = feats.sample(frac=1.0, random_state=0)
    ok = D.keyed_merge(shuffled, parts)
    got = ok.set_index("subject_id")["f"].sort_index()
    assert (got.to_numpy() == np.arange(88.0)).all()  # feature row i still belongs to subject i+1
    bad = D.positional_merge(shuffled, parts)
    assert (bad["participant_id_x"] != bad["participant_id_y"]).sum() > 50


def test_keyed_merge_aborts_on_missing_or_duplicate():
    parts = _participants()
    feats = pd.DataFrame({"participant_id": parts["participant_id"][:87], "f": 0.0})
    with pytest.raises(D.LabelAlignmentError):
        D.keyed_merge(feats, parts)
    dup = pd.DataFrame({"participant_id": list(parts["participant_id"]) + ["sub-001"], "f": 0.0})
    with pytest.raises(D.LabelAlignmentError):
        D.keyed_merge(dup, parts)


def test_qeeg28_and_theta_sensitivity():
    rng = np.random.default_rng(0)
    sf, T = 250.0, 250 * 60
    t = np.arange(T) / sf
    base = rng.normal(0, 1, (19, T))
    slow = base + 3 * np.sin(2 * np.pi * 6 * t)
    f0, names = qeeg28(base, sf, D.CHANNELS_19)
    f1, _ = qeeg28(slow, sf, D.CHANNELS_19)
    assert len(f0) == len(names) == 28 and np.isfinite(f0).all()
    i = names.index("theta_alpha_ratio")
    assert f1[i] > 5 * f0[i]


def test_epoch_slowing_features_dimension():
    freqs = np.linspace(0.5, 30, 60)
    psds = np.random.default_rng(0).random((3, 19, 60))
    F = epoch_slowing_features(psds, freqs, BANDS_NOTEBOOK)
    assert F.shape == (3, 95)
    assert np.allclose(F[:, 1:5].sum(1), 1, atol=1e-6)  # relative powers of channel 1 sum to one


def test_gsp_features():
    x = np.random.default_rng(0).normal(size=(19, 2000))
    f = gsp_features(x)
    assert len(f) == len(GSP_FEATURES) == 11 and np.isfinite(f).all()
    d = dict(zip(GSP_FEATURES, f))
    assert np.isclose(d["signal_power"], d["signal_energy"] / 2000)
    assert 0 < d["stationary_ratio"] <= 1
