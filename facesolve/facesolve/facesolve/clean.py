"""Clean raw landmark tracks: reject outliers, fill occlusion gaps, and remove
rigid head motion (Kabsch / orthogonal Procrustes on the head markers)."""
from __future__ import annotations

import numpy as np

from .rig import STABLE


def kabsch(P: np.ndarray, Q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Best rotation R and translation t with R @ p + t ~= q (least squares).

    P, Q: (n, 3) corresponding points. Uses the SVD of the cross-covariance
    and flips the last singular vector if needed so det(R) = +1 (no reflection).
    """
    pc, qc = P.mean(0), Q.mean(0)
    H = (P - pc).T @ (Q - qc)
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    D = np.diag([1.0, 1.0, d])
    R = Vt.T @ D @ U.T
    return R, qc - R @ pc


def _nanmedian_filter(x: np.ndarray, half: int) -> np.ndarray:
    """Temporal median over a +/-half window, ignoring NaNs. x: (T, ...)"""
    T = len(x)
    pad = np.concatenate([np.full((half,) + x.shape[1:], np.nan), x,
                          np.full((half,) + x.shape[1:], np.nan)])
    win = np.stack([pad[i : i + T] for i in range(2 * half + 1)])
    return np.nanmedian(win, axis=0)


def reject_outliers(raw: np.ndarray, thresh_mm: float = 8.0, half: int = 2) -> np.ndarray:
    """Return a (T, L) mask of samples that jump away from their temporal median."""
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN windows
        med = _nanmedian_filter(raw, half)
    dev = np.linalg.norm(raw - med, axis=-1)
    return np.nan_to_num(dev, nan=0.0) > thresh_mm


def fill_gaps(x: np.ndarray) -> np.ndarray:
    """Linear interpolation over time for NaN samples (nearest value at the ends)."""
    x = x.copy()
    T = len(x)
    t = np.arange(T)
    flat = x.reshape(T, -1)
    for j in range(flat.shape[1]):
        bad = np.isnan(flat[:, j])
        if bad.all():
            flat[:, j] = 0.0
        elif bad.any():
            flat[bad, j] = np.interp(t[bad], t[~bad], flat[~bad, j])
    return flat.reshape(x.shape)


def stabilize(x: np.ndarray, neutral_lm: np.ndarray) -> np.ndarray:
    """Map every frame into head space by aligning the head markers to neutral."""
    out = np.empty_like(x)
    for t in range(len(x)):
        R, tr = kabsch(x[t, STABLE], neutral_lm[STABLE])
        out[t] = x[t] @ R.T + tr
    return out


def clean_sequence(raw: np.ndarray, neutral_lm: np.ndarray):
    """Full cleaning pass for one (T, L, 3) take. Returns (cleaned, outlier_mask)."""
    mask = reject_outliers(raw)
    x = raw.copy()
    x[mask] = np.nan
    x = fill_gaps(x)
    return stabilize(x, neutral_lm), mask
