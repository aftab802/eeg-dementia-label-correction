"""Quantitative-EEG spectral features."""
from __future__ import annotations

import numpy as np
from scipy.signal import welch

BANDS_4 = {"delta": (0.5, 4.0), "theta": (4.0, 8.0), "alpha": (8.0, 13.0), "beta": (13.0, 30.0)}
# Band edges used in the original cost-sensitive XGBoost notebook (alpha 8-12, beta 12-30).
BANDS_NOTEBOOK = {"delta": (0.5, 4.0), "theta": (4.0, 8.0), "alpha": (8.0, 12.0), "beta": (12.0, 30.0)}
REGIONS = {
    "frontal": ["Fp1", "Fp2", "F3", "F4", "F7", "F8", "Fz"],
    "temporal": ["T3", "T4", "T5", "T6"],
    "parietal": ["P3", "P4", "Pz"],
}


def band_powers(data: np.ndarray, sfreq: float, bands=BANDS_4, nperseg_s: float = 4.0) -> dict:
    """Mean Welch PSD per band and channel. data: (channels, samples)."""
    f, psd = welch(data, fs=sfreq, nperseg=int(nperseg_s * sfreq), axis=-1)
    return {b: psd[:, (f >= lo) & (f <= hi)].mean(axis=1) for b, (lo, hi) in bands.items()}


def qeeg28(data: np.ndarray, sfreq: float, ch_names: list[str]) -> tuple[np.ndarray, list[str]]:
    """The 28 qEEG features of the label-correction paper, per subject.

    For each of the four bands (delta 0.5-4, theta 4-8, alpha 8-13, beta
    13-30 Hz): mean, SD and max of channel band power, and its frontal,
    temporal and parietal regional means (6 x 4 = 24); plus four cross-band
    ratios of mean band power: theta/alpha, delta/alpha, delta/theta,
    beta/alpha (24 + 4 = 28). The band power itself (Eq. 1 of the paper) enters
    as the per-band mean, so it is not counted twice.
    """
    bp = band_powers(data, sfreq)
    feats, names = [], []
    idx = {c: i for i, c in enumerate(ch_names)}
    for b, p in bp.items():
        vals = [p.mean(), p.std(), p.max()]
        for r, chans in REGIONS.items():
            sel = [idx[c] for c in chans if c in idx]
            vals.append(p[sel].mean() if sel else np.nan)
        feats += vals
        names += [f"{b}_{s}" for s in ("mean", "std", "max", "frontal", "temporal", "parietal")]
    m = {b: bp[b].mean() for b in bp}
    for a, c in (("theta", "alpha"), ("delta", "alpha"), ("delta", "theta"), ("beta", "alpha")):
        feats.append(m[a] / (m[c] + 1e-12)); names.append(f"{a}_{c}_ratio")
    return np.asarray(feats, dtype=float), names


def epoch_slowing_features(psds: np.ndarray, freqs: np.ndarray, bands=BANDS_NOTEBOOK) -> np.ndarray:
    """Per-epoch features of the original cost-sensitive XGBoost notebook.

    For every channel: slowing ratio (delta+theta)/(alpha+beta) followed by the
    four relative band powers -> 5 x n_channels (95 for 19 channels).
    psds: (epochs, channels, freqs).
    """
    out = np.empty((psds.shape[0], psds.shape[1] * 5))
    for ep in range(psds.shape[0]):
        row = []
        for ch in range(psds.shape[1]):
            p = {b: psds[ep, ch, (freqs >= lo) & (freqs <= hi)].mean() for b, (lo, hi) in bands.items()}
            row.append((p["delta"] + p["theta"]) / (p["alpha"] + p["beta"] + 1e-10))
            tot = sum(p.values()) + 1e-10
            row += [p[b] / tot for b in bands]
        out[ep] = row
    return out
