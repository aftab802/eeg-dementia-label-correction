"""OpenNeuro ds004504 (AD / FTD / HC, 88 subjects) access and label alignment.

Dataset: https://openneuro.org/datasets/ds004504 (v1.0.6), Miltiadous et al.
2023, DOI 10.3390/data8060095. Download with e.g.
``openneuro-py download --dataset ds004504`` or ``aws s3 sync --no-sign-request
s3://openneuro.org/ds004504 ds004504``.
"""
from __future__ import annotations

import os
import re

import numpy as np
import pandas as pd

CHANNELS_19 = ["Fp1", "Fp2", "F3", "F4", "C3", "C4", "P3", "P4", "O1", "O2",
               "F7", "F8", "T3", "T4", "T5", "T6", "Fz", "Cz", "Pz"]
GROUP_NAMES = {"A": "AD", "F": "FTD", "C": "HC"}


class LabelAlignmentError(AssertionError):
    """Raised when features and diagnostic labels cannot be joined one-to-one."""


def canonical_id(value) -> str:
    """Normalise identifiers such as 'sub-007', 'sub-007_task-eyesclosed_eeg.set',
    '7' or 7 to the BIDS form 'sub-007'."""
    s = str(value)
    m = re.search(r"sub-?(\d+)", s, flags=re.I) or re.search(r"(\d+)", s)
    if not m:
        raise ValueError(f"No subject number in {value!r}")
    return f"sub-{int(m.group(1)):03d}"


def load_participants(bids_root: str) -> pd.DataFrame:
    df = pd.read_csv(os.path.join(bids_root, "participants.tsv"), sep="\t")
    df["subject_id"] = df["participant_id"].map(canonical_id)
    df["diagnosis"] = df["Group"].map(GROUP_NAMES)
    return df


def positional_merge(features: pd.DataFrame, participants: pd.DataFrame) -> pd.DataFrame:
    """The ORIGINAL defect: join by row position. Kept only to demonstrate it."""
    return features.reset_index(drop=True).merge(participants.reset_index(drop=True),
                                                 left_index=True, right_index=True)


def keyed_merge(features: pd.DataFrame, participants: pd.DataFrame, id_col: str = "participant_id",
                expected_n: int | None = 88, one_row_per_subject: bool = True) -> pd.DataFrame:
    """Identifier-keyed merge with the verification of Algorithm 1 (label-correction paper).

    Aborts (raises LabelAlignmentError) unless every feature row receives exactly
    one non-null diagnosis. For epoch-level tables set ``one_row_per_subject=False``.
    """
    f = features.copy()
    f["subject_id"] = f[id_col].map(canonical_id)
    p = participants.drop(columns=["participant_id"], errors="ignore")
    if "subject_id" not in p:
        raise LabelAlignmentError("participants table needs a 'subject_id' column (use load_participants)")
    if p["subject_id"].duplicated().any():
        raise LabelAlignmentError("Duplicate subject identifiers in participants table")
    m = f.merge(p, on="subject_id", how="inner", validate="many_to_one")
    if len(m) != len(f):
        missing = sorted(set(f["subject_id"]) - set(p["subject_id"]))
        raise LabelAlignmentError(f"{len(f) - len(m)} feature rows without a label: {missing[:10]}")
    if m["Group"].isna().any():
        raise LabelAlignmentError("Null diagnostic label after merge")
    if one_row_per_subject:
        if m["subject_id"].duplicated().any():
            raise LabelAlignmentError("More than one feature row per subject")
        if expected_n is not None and not (len(m) == len(p) == expected_n):
            raise LabelAlignmentError(f"Expected {expected_n} subjects, got features={len(m)}, labels={len(p)}")
    return m


def derivative_recordings(bids_root: str, task: str | None = "eyesclosed"):
    """Yield (subject_id, path) for the denoised recordings in ``derivatives/``.

    ``task='eyesclosed'`` selects the ds004504 resting recordings. ``task=None``
    takes the single .set file of each subject whatever its task label, which is
    what the companion photic-stimulation dataset ds006036 needs.
    """
    d = os.path.join(bids_root, "derivatives")
    if not any(x.startswith("sub-") for x in os.listdir(d)):
        # ds006036 keeps its denoised files one level down, in derivatives/eeglab/
        for pipeline in sorted(os.listdir(d)):
            pd_ = os.path.join(d, pipeline)
            if os.path.isdir(pd_) and any(x.startswith("sub-") for x in os.listdir(pd_)):
                d = pd_
                break
    for sub in sorted(os.listdir(d)):
        if not sub.startswith("sub-"):
            continue
        eeg = os.path.join(d, sub, "eeg")
        if task is not None:
            path = os.path.join(eeg, f"{sub}_task-{task}_eeg.set")
            if os.path.exists(path):
                yield sub, path
        elif os.path.isdir(eeg):
            sets = sorted(f for f in os.listdir(eeg) if f.endswith("_eeg.set"))
            if sets:
                yield sub, os.path.join(eeg, sets[0])


def load_recording(path: str, l_freq: float = 0.5, h_freq: float = 30.0, preload: bool = True):
    import mne
    raw = mne.io.read_raw_eeglab(path, preload=preload, verbose=False)
    raw.pick([c for c in CHANNELS_19 if c in raw.ch_names])
    if preload:
        raw.filter(l_freq=l_freq, h_freq=h_freq, fir_design="firwin", verbose=False)
    return raw
