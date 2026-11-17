"""Synthetic data in two flavours, mirroring a real facial pipeline:

* raw capture  - temporally coherent performances seen through a moving head,
                 with sensor noise, occlusion gaps and marker swaps (outliers)
* artist set   - clean, curated reference poses (single shapes and pairs),
                 the kind of high-fidelity data artists tune by hand
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .rig import K, SHAPES, FaceRig, landmark_indices


def rotation(yaw, pitch, roll) -> np.ndarray:
    cy, sy = np.cos(yaw), np.sin(yaw)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cr, sr = np.cos(roll), np.sin(roll)
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rx = np.array([[1, 0, 0], [0, cp, -sp], [0, sp, cp]])
    Rz = np.array([[cr, -sr, 0], [sr, cr, 0], [0, 0, 1]])
    return Rz @ Rx @ Ry


def _smooth_keys(rng, T, dims, every, lo, hi):
    """Piecewise-linear random curves through keyframes every `every` frames."""
    n = T // every + 2
    keys = rng.uniform(lo, hi, size=(n, dims))
    t = np.arange(T) / every
    i = np.floor(t).astype(int)
    f = (t - i)[:, None]
    return keys[i] * (1 - f) + keys[i + 1] * f


def sample_weights(rng, T) -> np.ndarray:
    """Sparse, smooth, roughly symmetric performance weights (T, K)."""
    w = _smooth_keys(rng, T, K, 12, -0.6, 1.0).clip(0, 1)  # many shapes sit at 0
    for k, name in enumerate(SHAPES):  # left/right coupling, as faces do
        if name.endswith("_L"):
            r = SHAPES.index(name[:-2] + "_R")
            mix = rng.uniform(0.3, 0.9)
            w[:, r] = (mix * w[:, k] + (1 - mix) * w[:, r]).clip(0, 1)
    return w


@dataclass
class Capture:
    weights: np.ndarray  # (T, K) ground truth
    clean: np.ndarray  # (T, L, 3) landmarks in head space (ground truth)
    raw: np.ndarray  # (T, L, 3) what the tracker reports (NaN = occluded)
    outlier_mask: np.ndarray  # (T, L) True where a marker swap was injected


def simulate_capture(
    rig: FaceRig,
    n_seq: int,
    T: int = 120,
    noise_mm: float = 0.4,
    gap_rate: float = 0.01,
    outlier_rate: float = 0.005,
    seed: int = 0,
) -> Capture:
    rng = np.random.default_rng(seed)
    idx = landmark_indices()
    Wl, Cl, Rl, Ol = [], [], [], []
    for _ in range(n_seq):
        w = sample_weights(rng, T)
        clean = rig.forward(w)[:, idx]  # (T, L, 3)
        ang = np.deg2rad(_smooth_keys(rng, T, 3, 30, -1, 1) * [15, 10, 5])
        trans = _smooth_keys(rng, T, 3, 30, -20, 20)
        raw = np.empty_like(clean)
        for t in range(T):
            R = rotation(*ang[t])
            raw[t] = clean[t] @ R.T + trans[t]
        raw += rng.normal(0, noise_mm, raw.shape)
        L = raw.shape[1]
        # occlusion gaps: runs of 3-10 missing frames
        for m in range(L):
            for t in np.flatnonzero(rng.random(T) < gap_rate):
                raw[t : t + rng.integers(3, 11), m] = np.nan
        # marker swaps / tracker jumps: 15-30 mm spikes
        out = (rng.random((T, L)) < outlier_rate) & ~np.isnan(raw[..., 0])
        d = rng.normal(size=(out.sum(), 3))
        d *= (rng.uniform(15, 30, len(d)) / np.linalg.norm(d, axis=1))[:, None]
        raw[out] += d
        Wl.append(w), Cl.append(clean), Rl.append(raw), Ol.append(out)
    return Capture(*(np.stack(x) for x in (Wl, Cl, Rl, Ol)))


def artist_set(rig: FaceRig, steps=(0.25, 0.5, 0.75, 1.0)) -> tuple[np.ndarray, np.ndarray]:
    """Curated poses: every shape alone at 4 levels, plus every pair at 0.5/1.0."""
    W = [np.zeros(K)]
    for k in range(K):
        for s in steps:
            w = np.zeros(K)
            w[k] = s
            W.append(w)
    for a in range(K):
        for b in range(a + 1, K):
            for s in (0.5, 1.0):
                w = np.zeros(K)
                w[[a, b]] = s
                W.append(w)
    W = np.asarray(W)
    return W, rig.forward(W)[:, landmark_indices()]
