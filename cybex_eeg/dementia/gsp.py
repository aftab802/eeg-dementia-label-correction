"""Graph-signal-processing (GSP) descriptors of multichannel EEG.

The label-correction paper retains eleven graph-spectral features from the
open-source pipeline of Mootoo et al. (medRxiv 2023, doi
10.1101/2023.11.01.23297940), which follows Mathur & Chakka (SPIN 2020). When
that pipeline's feature table is available, load it and pass it through
``ds004504.keyed_merge``; that is what the paper did. This module is an
independent reimplementation for users without the table. Definitions:

graph      W = |Pearson correlation| between channels, zero diagonal; L = D - W
signal     x = per-channel signal power (time-averaged squared amplitude)
1 stationary_ratio   ||diag(U^T C U)||_2 / ||U^T C U||_F, C = channel covariance
2 tikhonov_norm      ||(I + tau L)^-1 x||_2 (graph-Tikhonov smoothed signal), tau = 1
3 total_variation    x^T L x
4 graph_energy       sum |eig(W)|
5 spectral_entropy   Shannon entropy (bits) of normalised |U^T x|^2
6 signal_energy      sum over channels and samples of s^2
7 signal_power       signal_energy / number of samples
8 unique_clusters    connected components of the graph W > threshold
9 average_degree     mean binary degree of W > threshold
10 heat_trace        trace(exp(-t L)) = sum exp(-t lambda), t = 1
11 diffusion_distance mean ||exp(-t L)(e_i - e_j)||_2 over channel pairs, t = 1
Numbers from this reimplementation are not expected to equal those of the
original table exactly.
"""
from __future__ import annotations

import numpy as np
from scipy.sparse.csgraph import connected_components

GSP_FEATURES = ["stationary_ratio", "tikhonov_norm", "total_variation", "graph_energy",
                "spectral_entropy", "signal_energy", "signal_power", "unique_clusters",
                "average_degree", "heat_trace", "diffusion_distance"]


def gsp_features(data: np.ndarray, threshold: float = 0.5, tau: float = 1.0, t: float = 1.0) -> np.ndarray:
    """Eleven GSP descriptors for one recording. data: (channels, samples)."""
    data = np.asarray(data, dtype=float)
    n = data.shape[0]
    W = np.abs(np.corrcoef(data))
    np.fill_diagonal(W, 0.0)
    W = np.nan_to_num(W)
    Lap = np.diag(W.sum(1)) - W
    lam, U = np.linalg.eigh(Lap)
    x = (data ** 2).mean(axis=1)
    C = np.cov(data)
    S = U.T @ C @ U
    stationary = np.linalg.norm(np.diag(S)) / (np.linalg.norm(S) + 1e-12)
    tikh = np.linalg.norm(np.linalg.solve(np.eye(n) + tau * Lap, x))
    tv = float(x @ Lap @ x)
    genergy = float(np.abs(np.linalg.eigvalsh(W)).sum())
    xh = (U.T @ x) ** 2
    p = xh / (xh.sum() + 1e-12)
    sent = float(-(p[p > 0] * np.log2(p[p > 0])).sum())
    senergy = float((data ** 2).sum())
    spower = senergy / data.shape[1]
    A = (W > threshold).astype(int)
    nclust = connected_components(A, directed=False)[0]
    avgdeg = float(A.sum(1).mean())
    H = U @ np.diag(np.exp(-t * lam)) @ U.T
    htrace = float(np.exp(-t * lam).sum())
    dd = [np.linalg.norm(H[:, i] - H[:, j]) for i in range(n) for j in range(i + 1, n)]
    return np.array([stationary, tikh, tv, genergy, sent, senergy, spower, nclust, avgdeg, htrace,
                     float(np.mean(dd))])
