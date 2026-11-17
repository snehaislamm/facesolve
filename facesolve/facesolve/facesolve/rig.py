"""Procedural face rig: template mesh, artist-style blendshapes, a rotational
jaw and FACS-style combination correctives.

The rig is intentionally *nonlinear* (jaw is a true rotation, correctives fire
on products of weights) so that a linear blendshape solve cannot fit it exactly,
the same reason production rigs need learned solvers.

Units: millimetres. Axes: +X subject's left, +Y up, +Z toward camera.
"""
from __future__ import annotations

import numpy as np

GRID = 40  # 40 x 40 = 1,600 vertices

SHAPES = [
    "jawOpen",
    "browInnerUp_L", "browInnerUp_R",
    "browDown_L", "browDown_R",
    "eyeBlink_L", "eyeBlink_R",
    "mouthSmile_L", "mouthSmile_R",
    "mouthFrown_L", "mouthFrown_R",
    "cheekPuff_L", "cheekPuff_R",
    "mouthPucker", "mouthFunnel",
    "noseSneer",
]
K = len(SHAPES)
JAW = 0

# (shape name, centre (u, v), radius, displacement direction xyz, magnitude mm)
_REGIONS = {
    "browInnerUp_L": ((0.22, 0.52), 0.18, (0, 1, 0.2), 6.0),
    "browInnerUp_R": ((-0.22, 0.52), 0.18, (0, 1, 0.2), 6.0),
    "browDown_L": ((0.30, 0.48), 0.22, (-0.3, -1, 0.3), 5.0),
    "browDown_R": ((-0.30, 0.48), 0.22, (0.3, -1, 0.3), 5.0),
    "eyeBlink_L": ((0.32, 0.30), 0.10, (0, -1, 0.1), 7.0),
    "eyeBlink_R": ((-0.32, 0.30), 0.10, (0, -1, 0.1), 7.0),
    "mouthSmile_L": ((0.30, -0.45), 0.16, (0.7, 0.7, -0.2), 9.0),
    "mouthSmile_R": ((-0.30, -0.45), 0.16, (-0.7, 0.7, -0.2), 9.0),
    "mouthFrown_L": ((0.28, -0.50), 0.15, (0.2, -1, 0), 6.0),
    "mouthFrown_R": ((-0.28, -0.50), 0.15, (-0.2, -1, 0), 6.0),
    "cheekPuff_L": ((0.50, -0.20), 0.25, (0.4, 0, 1), 8.0),
    "cheekPuff_R": ((-0.50, -0.20), 0.25, (-0.4, 0, 1), 8.0),
    "mouthPucker": ((0.0, -0.45), 0.18, (0, 0, 1), 8.0),
    "mouthFunnel": ((0.0, -0.45), 0.14, (0, 0, 1), 5.0),
    "noseSneer": ((0.0, -0.02), 0.08, (0, 1, 0.3), 4.0),
}

# combination correctives: (shape a, shape b, centre, radius, dir, mag)
_CORRECTIVES = [
    ("mouthSmile_L", "jawOpen", (0.28, -0.52), 0.14, (0.3, 1, 0.4), 5.0),
    ("mouthSmile_R", "jawOpen", (-0.28, -0.52), 0.14, (-0.3, 1, 0.4), 5.0),
    ("eyeBlink_L", "mouthSmile_L", (0.32, 0.22), 0.12, (0, 1, 0.3), 3.0),
    ("eyeBlink_R", "mouthSmile_R", (-0.32, 0.22), 0.12, (0, 1, 0.3), 3.0),
    ("mouthPucker", "jawOpen", (0.0, -0.50), 0.16, (0, 0.6, -1), 4.0),
]

JAW_MAX_DEG = 15.0
JAW_PIVOT = np.array([0.0, -5.0, -40.0])  # behind the face, near the ears


def _uv() -> tuple[np.ndarray, np.ndarray]:
    u, v = np.meshgrid(np.linspace(-1, 1, GRID), np.linspace(-1, 1, GRID))
    return u.ravel(), v.ravel()


def template() -> tuple[np.ndarray, np.ndarray]:
    """Return neutral vertices (V, 3) and triangle faces (F, 3)."""
    u, v = _uv()
    x = 75.0 * u
    y = 100.0 * v
    z = 55.0 * np.sqrt(np.clip(1 - 0.75 * u**2 - 0.45 * v**2, 0, None))
    z += 18.0 * np.exp(-((u / 0.12) ** 2 + ((v - 0.0) / 0.22) ** 2))  # nose
    verts = np.stack([x, y, z], 1)
    faces = []
    for r in range(GRID - 1):
        for c in range(GRID - 1):
            i = r * GRID + c
            faces += [(i, i + 1, i + GRID), (i + 1, i + GRID + 1, i + GRID)]
    return verts, np.asarray(faces, dtype=np.int32)


def _falloff(centre, radius) -> np.ndarray:
    u, v = _uv()
    d2 = (u - centre[0]) ** 2 + (v - centre[1]) ** 2
    return np.exp(-d2 / (2 * radius**2))


def _delta(centre, radius, direction, mag) -> np.ndarray:
    d = np.asarray(direction, float)
    d /= np.linalg.norm(d)
    return _falloff(centre, radius)[:, None] * d[None, :] * mag


def _jaw_mask() -> np.ndarray:
    u, v = _uv()
    # smooth step: 0 above v=-0.05, 1 below v=-0.30, fades out at the sides
    t = np.clip((-0.05 - v) / 0.25, 0, 1)
    side = np.exp(-(u / 0.85) ** 6)
    return (t * t * (3 - 2 * t)) * side


class FaceRig:
    """Nonlinear forward model: weights (K,) or (N, K) -> vertices."""

    def __init__(self) -> None:
        self.neutral, self.faces = template()
        self.V = len(self.neutral)
        self.deltas = np.zeros((K, self.V, 3))
        for k, name in enumerate(SHAPES):
            if name in _REGIONS:
                self.deltas[k] = _delta(*_REGIONS[name])
        self.jaw_mask = _jaw_mask()
        self.correctives = [
            (SHAPES.index(a), SHAPES.index(b), _delta(c, r, d, m))
            for a, b, c, r, d, m in _CORRECTIVES
        ]
        # the artist's *linear* approximation of the jaw: its full-open pose
        self.deltas[JAW] = self._jaw(np.ones(1))[0]

    def _jaw(self, w: np.ndarray) -> np.ndarray:
        """True rotational jaw displacement for weights w (N,) -> (N, V, 3)."""
        ang = np.deg2rad(JAW_MAX_DEG) * w[:, None] * self.jaw_mask[None, :]  # (N,V)
        p = self.neutral - JAW_PIVOT
        c, s = np.cos(ang), np.sin(ang)
        y = c * p[None, :, 1] - s * p[None, :, 2]
        z = s * p[None, :, 1] + c * p[None, :, 2]
        out = np.zeros(ang.shape + (3,))
        out[..., 1] = y - p[None, :, 1]
        out[..., 2] = z - p[None, :, 2]
        return out

    def forward(self, w: np.ndarray) -> np.ndarray:
        w = np.atleast_2d(np.asarray(w, float))
        lin = np.einsum("nk,kvd->nvd", w[:, 1:], self.deltas[1:])
        out = self.neutral[None] + lin + self._jaw(w[:, JAW])
        for a, b, d in self.correctives:
            out += (w[:, a] * w[:, b])[:, None, None] * d[None]
        return out

    def forward_linear(self, w: np.ndarray) -> np.ndarray:
        """What a plain linear blendshape rig (e.g. UsdSkel, Maya blendShape) plays back."""
        w = np.atleast_2d(np.asarray(w, float))
        return self.neutral[None] + np.einsum("nk,kvd->nvd", w, self.deltas)


# ---- 68 landmarks, laid out like the common iBUG/dlib 68-point scheme --------
def _ring(cx, cy, rx, ry, n, a0=0.0, a1=2 * np.pi, endpoint=False):
    a = np.linspace(a0, a1, n, endpoint=endpoint)
    return [(cx + rx * np.cos(t), cy + ry * np.sin(t)) for t in a]


def landmark_uv() -> list[tuple[float, float]]:
    pts = []
    pts += _ring(0, -0.05, 0.78, 0.80, 17, np.pi * 1.05, np.pi * 1.95, True)  # jaw 0-16
    pts += [(-0.50 + 0.07 * i, 0.50 + 0.03 * np.sin(i)) for i in range(5)]  # R brow 17-21
    pts += [(0.22 + 0.07 * i, 0.50 + 0.03 * np.sin(i)) for i in range(5)]  # L brow 22-26
    pts += [(0, 0.22 - 0.07 * i) for i in range(4)]  # nose bridge 27-30
    pts += [(-0.10 + 0.05 * i, -0.08) for i in range(5)]  # nostrils 31-35
    pts += _ring(-0.32, 0.30, 0.10, 0.04, 6)  # R eye 36-41
    pts += _ring(0.32, 0.30, 0.10, 0.04, 6)  # L eye 42-47
    pts += _ring(0, -0.45, 0.26, 0.10, 12)  # outer lip 48-59
    pts += _ring(0, -0.45, 0.16, 0.05, 8)  # inner lip 60-67
    # 4 rigid head markers (forehead + temples), as on a capture helmet: 68-71
    pts += [(-0.30, 0.95), (0.30, 0.95), (-0.92, 0.35), (0.92, 0.35)]
    return pts


def landmark_indices() -> np.ndarray:
    u, v = _uv()
    idx = []
    for pu, pv in landmark_uv():
        d = (u - pu) ** 2 + (v - pv) ** 2
        d[idx] = np.inf  # keep indices unique
        idx.append(int(np.argmin(d)))
    return np.asarray(idx)


# markers that ride the skull, not the expression (capture-helmet head markers)
STABLE = np.array([68, 69, 70, 71])
L = 72
