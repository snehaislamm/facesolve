"""Landmarks -> blendshape weights.

* MLPSolver: a small neural network (2 hidden layers, ReLU, sigmoid output so
  weights stay in [0, 1]) trained with Adam. Written in NumPy with hand-derived
  backprop so it runs anywhere; `scripts/train_torch.py` is the PyTorch version
  of the same network.
* LinearSolver: the classic baseline, a bounded least-squares fit of the
  linear blendshape basis to the landmark offsets.
"""
from __future__ import annotations

import json

import numpy as np
from scipy.optimize import lsq_linear


def features(lm: np.ndarray, neutral_lm: np.ndarray) -> np.ndarray:
    """(N, L, 3) landmarks -> (N, 3L) offsets from neutral."""
    return (lm - neutral_lm[None]).reshape(len(lm), -1)


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


class MLPSolver:
    def __init__(self, n_in: int, n_out: int, hidden=(256, 256), seed: int = 0):
        rng = np.random.default_rng(seed)
        sizes = (n_in, *hidden, n_out)
        self.W = [rng.normal(0, np.sqrt(2 / a), (a, b)) for a, b in zip(sizes[:-1], sizes[1:])]
        self.b = [np.zeros(b) for b in sizes[1:]]
        self.mu = np.zeros(n_in)
        self.sd = np.ones(n_in)

    # ---------- forward / backward ----------
    def _forward(self, X):
        acts = [(X - self.mu) / self.sd]
        for i, (W, b) in enumerate(zip(self.W, self.b)):
            z = acts[-1] @ W + b
            acts.append(np.maximum(z, 0) if i < len(self.W) - 1 else _sigmoid(z))
        return acts

    def predict(self, X, dtype=np.float64):
        h = ((X - self.mu) / self.sd).astype(dtype)
        for i, (W, b) in enumerate(zip(self.W, self.b)):
            h = h @ W.astype(dtype) + b.astype(dtype)
            h = np.maximum(h, 0) if i < len(self.W) - 1 else _sigmoid(h)
        return h

    def loss_and_grads(self, X, Y):
        acts = self._forward(X)
        P = acts[-1]
        n = len(X)
        loss = np.mean((P - Y) ** 2)
        g = 2 * (P - Y) / (n * Y.shape[1]) * P * (1 - P)  # through the sigmoid
        gW, gb = [None] * len(self.W), [None] * len(self.W)
        for i in reversed(range(len(self.W))):
            gW[i] = acts[i].T @ g
            gb[i] = g.sum(0)
            if i:
                g = (g @ self.W[i].T) * (acts[i] > 0)
        return loss, gW, gb

    # ---------- training ----------
    def fit(self, X, Y, Xval=None, Yval=None, epochs=40, batch=256, lr=1e-3,
            noise_std=0.0, seed=0, log=print):
        rng = np.random.default_rng(seed)
        self.mu, self.sd = X.mean(0), X.std(0) + 1e-6
        params = self.W + self.b
        m = [np.zeros_like(p) for p in params]
        v = [np.zeros_like(p) for p in params]
        b1, b2, eps, step = 0.9, 0.999, 1e-8, 0
        hist = []
        for ep in range(epochs):
            lr_ep = lr * 0.5 * (1 + np.cos(np.pi * ep / epochs))  # cosine decay
            order = rng.permutation(len(X))
            tot = 0.0
            for s in range(0, len(X), batch):
                j = order[s : s + batch]
                xb = X[j] + rng.normal(0, noise_std, X[j].shape) if noise_std else X[j]
                loss, gW, gb = self.loss_and_grads(xb, Y[j])
                tot += loss * len(j)
                step += 1
                for k, (p, g) in enumerate(zip(params, gW + gb)):
                    m[k] = b1 * m[k] + (1 - b1) * g
                    v[k] = b2 * v[k] + (1 - b2) * g * g
                    mh, vh = m[k] / (1 - b1**step), v[k] / (1 - b2**step)
                    p -= lr_ep * mh / (np.sqrt(vh) + eps)
            rec = {"epoch": ep + 1, "train_mse": tot / len(X)}
            if Xval is not None:
                rec["val_mse"] = float(np.mean((self.predict(Xval) - Yval) ** 2))
            hist.append(rec)
            if log and (ep % 5 == 4 or ep == 0):
                log(json.dumps({k: round(v, 6) if isinstance(v, float) else v for k, v in rec.items()}))
        return hist

    def save(self, path):
        np.savez(path, mu=self.mu, sd=self.sd,
                 **{f"W{i}": W for i, W in enumerate(self.W)},
                 **{f"b{i}": b for i, b in enumerate(self.b)})

    @classmethod
    def load(cls, path):
        d = np.load(path)
        n = sum(1 for k in d.files if k.startswith("W"))
        obj = cls.__new__(cls)
        obj.W = [d[f"W{i}"] for i in range(n)]
        obj.b = [d[f"b{i}"] for i in range(n)]
        obj.mu, obj.sd = d["mu"], d["sd"]
        return obj


class LinearSolver:
    """Bounded least squares: min ||B w - x||^2 s.t. 0 <= w <= 1."""

    def __init__(self, rig, lm_idx):
        self.B = rig.deltas[:, lm_idx].reshape(len(rig.deltas), -1).T  # (3L, K)

    def predict(self, X):
        return np.stack([lsq_linear(self.B, x, bounds=(0, 1), method="bvls").x for x in X])
