"""Run the command-line scripts end to end on a small synthetic BIDS / EDF cohort."""
import json
import os
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

mne = pytest.importorskip("mne")
pytest.importorskip("eeglabio")
SCRIPTS = os.path.join(os.path.dirname(__file__), "..", "scripts")
CH = ["Fp1", "Fp2", "F3", "F4", "C3", "C4", "P3", "P4", "O1", "O2", "F7", "F8", "T3", "T4", "T5", "T6", "Fz", "Cz", "Pz"]


def _raw(group, rng, sf=128.0, seconds=40, amp=1.0):
    t = np.arange(int(sf * seconds)) / sf
    x = rng.normal(0, 10e-6, (19, t.size))
    slow = {"A": 6.0, "F": 9.0, "C": 11.0}[group]
    x += amp * 15e-6 * np.sin(2 * np.pi * slow * t)
    return mne.io.RawArray(x, mne.create_info(CH, sf, "eeg"), verbose=False)


@pytest.fixture
def bids(tmp_path):
    rng = np.random.default_rng(0)
    groups = ["A"] * 6 + ["F"] * 5 + ["C"] * 6
    rows = []
    for i, g in enumerate(groups, 1):
        sub = f"sub-{i:03d}"
        d = tmp_path / "derivatives" / sub / "eeg"
        d.mkdir(parents=True)
        mne.export.export_raw(str(d / f"{sub}_task-eyesclosed_eeg.set"), _raw(g, rng), fmt="eeglab", verbose=False)
        rows.append({"participant_id": sub, "Gender": "F", "Age": 65, "Group": g, "MMSE": 20})
    pd.DataFrame(rows).to_csv(tmp_path / "participants.tsv", sep="\t", index=False)
    return tmp_path


def _run(script, *args, cwd):
    r = subprocess.run([sys.executable, os.path.join(SCRIPTS, script), *args, "--outdir", str(cwd / "res")],
                       cwd=cwd, capture_output=True, text=True, timeout=900)
    assert r.returncode == 0, r.stderr[-3000:]
    files = sorted((cwd / "res").glob("*.json"))
    return json.loads(files[-1].read_text())


@pytest.mark.skipif(not os.path.exists(os.path.join(SCRIPTS, "run_label_correction.py")), reason="not in this repo")
def test_label_correction_script(bids, tmp_path):
    r = _run("run_label_correction.py", "--bids-root", str(bids), "--quick-grid", "--expected-n", "17", "--features-out",
             str(tmp_path / "f.csv"), cwd=tmp_path)
    assert r["n_features"] == 39 and r["n_subjects"] == 17
    assert r["rows_misassigned_by_positional_join"] == 0
    assert "nested_cv_mean" in r["xgb_gridsearch"]


@pytest.mark.skipif(not os.path.exists(os.path.join(SCRIPTS, "run_subject_vote_xgb.py")), reason="not in this repo")
def test_subject_vote_script(bids, tmp_path):
    r = _run("run_subject_vote_xgb.py", "--bids-root", str(bids), cwd=tmp_path)
    assert len(r["groupkfold"]["subjects"]) == 17 and 0 <= r["holdout_accuracy"] <= 1


@pytest.mark.skipif(not os.path.exists(os.path.join(SCRIPTS, "run_lead_eeg.py")), reason="not in this repo")
def test_lead_script_with_labels_csv(tmp_path):
    rng = np.random.default_rng(1)
    rows = []
    for i in range(30):
        lab = int(i < 20)
        raw = _raw("A" if lab else "C", rng, seconds=20, amp=2.0 if lab else 1.0)
        p = tmp_path / f"s{i:02d}.edf"
        mne.export.export_raw(str(p), raw, fmt="edf", verbose=False) if hasattr(mne, "export") else None
        rows.append({"path": str(p), "label": lab})
    pd.DataFrame(rows).to_csv(tmp_path / "labels.csv", index=False)
    r = _run("run_lead_eeg.py", "--labels", str(tmp_path / "labels.csv"), cwd=tmp_path)
    assert r["n_subjects"] == 30 and r["accuracy_pooled"] > 0.6


@pytest.mark.skipif(not os.path.exists(os.path.join(SCRIPTS, "run_lead_eeg.py")), reason="not in this repo")
def test_lead_script_ds004504_proxy(bids, tmp_path):
    r = _run("run_lead_eeg.py", "--ds004504", str(bids), "--k", "10", "--folds", "3", cwd=tmp_path)
    assert r["n_subjects"] == 12 and r["cohort"].startswith("ds004504")


@pytest.mark.skipif(not os.path.exists(os.path.join(SCRIPTS, "run_lead_cohort176.py")), reason="not in this repo")
def test_lead_cohort176_script(bids, tmp_path):
    photic = tmp_path / "ds006036"
    rng = np.random.default_rng(1)
    parts = pd.read_csv(bids / "participants.tsv", sep="\t")
    for sub, g in zip(parts.participant_id, parts.Group):
        d = photic / "derivatives" / sub / "eeg"
        d.mkdir(parents=True)
        mne.export.export_raw(str(d / f"{sub}_task-photomark_eeg.set"), _raw(g, rng), fmt="eeglab", verbose=False)
    parts.to_csv(photic / "participants.tsv", sep="\t", index=False)
    r = _run("run_lead_cohort176.py", "--ds004504", str(bids), "--ds006036", str(photic), "--k", "10", "--folds", "3",
             cwd=tmp_path)
    assert r["n_recordings"] == 34 and r["n_participants"] == 17 and r["n_dementia_recordings"] == 22
    g = r["runs"]["per_channel_deg2__participant_grouped"]
    assert g["grouped"] and g["participants_shared_across_folds"] == 0
    assert set(r["runs"]) == {f"{c}__{p}" for c in ("per_channel_deg2", "channel_average_deg4")
                              for p in ("recording_level", "participant_grouped")}
