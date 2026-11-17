import re
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from facesolve.capture import artist_set, rotation, simulate_capture  # noqa: E402
from facesolve.clean import clean_sequence, fill_gaps, kabsch  # noqa: E402
from facesolve.model import LinearSolver, MLPSolver, features  # noqa: E402
from facesolve.rig import K, SHAPES, STABLE, FaceRig, landmark_indices  # noqa: E402
from facesolve.usd_export import write_usda  # noqa: E402


@pytest.fixture(scope="module")
def rig():
    return FaceRig()


def test_neutral_pose_is_identity(rig):
    assert np.allclose(rig.forward(np.zeros(K))[0], rig.neutral)


def test_shapes_are_local_and_stable_markers_barely_move(rig):
    idx = landmark_indices()
    d = np.linalg.norm(rig.forward(np.eye(K)) - rig.neutral, axis=-1)
    assert (d.max(1) > 3).all()  # every shape does something visible
    assert d[:, idx[STABLE]].max() < 1.0  # head markers stay put (< 1 mm)


def test_rig_is_nonlinear(rig):
    w = np.zeros((1, K))
    w[0, SHAPES.index("jawOpen")] = 0.5
    w[0, SHAPES.index("mouthSmile_L")] = 1.0
    gap = np.abs(rig.forward(w) - rig.forward_linear(w)).max()
    assert gap > 1.0  # corrective + jaw rotation are not in the linear basis


def test_landmarks_unique():
    idx = landmark_indices()
    assert len(set(idx.tolist())) == len(idx) == 72


def test_kabsch_recovers_rigid_transform():
    rng = np.random.default_rng(0)
    P = rng.normal(size=(10, 3))
    R = rotation(0.3, -0.2, 0.1)
    t = np.array([5.0, -2.0, 1.0])
    Rh, th = kabsch(P, P @ R.T + t)
    assert np.allclose(Rh, R, atol=1e-9) and np.allclose(th, t, atol=1e-9)
    assert np.isclose(np.linalg.det(Rh), 1.0)


def test_fill_gaps_is_linear_interp():
    x = np.array([0.0, np.nan, np.nan, 3.0])[:, None]
    assert np.allclose(fill_gaps(x).ravel(), [0, 1, 2, 3])


def test_cleaning_removes_head_motion_and_outliers(rig):
    cap = simulate_capture(rig, 2, seed=7)
    nl = rig.neutral[landmark_indices()]
    for raw, gt, om in zip(cap.raw, cap.clean, cap.outlier_mask):
        clean, mask = clean_sequence(raw, nl)
        assert not np.isnan(clean).any()
        assert np.sqrt(np.mean(np.sum((clean - gt) ** 2, -1))) < 2.0  # mm
        if om.any():
            assert (mask & om).sum() / om.sum() > 0.9


def test_mlp_gradients_match_finite_differences():
    rng = np.random.default_rng(1)
    m = MLPSolver(6, 3, hidden=(5,), seed=1)
    X, Y = rng.normal(size=(4, 6)), rng.random((4, 3))
    _, gW, _ = m.loss_and_grads(X, Y)
    eps = 1e-6
    for (i, j) in [(0, 0), (3, 2), (5, 4)]:
        m.W[0][i, j] += eps
        lp = m.loss_and_grads(X, Y)[0]
        m.W[0][i, j] -= 2 * eps
        lm = m.loss_and_grads(X, Y)[0]
        m.W[0][i, j] += eps
        assert np.isclose(gW[0][i, j], (lp - lm) / (2 * eps), rtol=1e-4, atol=1e-9)


def test_mlp_outputs_valid_weights_and_learns(rig):
    W, L = artist_set(rig)
    nl = rig.neutral[landmark_indices()]
    X = features(L, nl)
    m = MLPSolver(X.shape[1], K, hidden=(64,))
    hist = m.fit(X, W, epochs=30, batch=64, lr=3e-3, log=None)
    assert hist[-1]["train_mse"] < hist[0]["train_mse"] / 3
    p = m.predict(X)
    assert p.min() >= 0 and p.max() <= 1


def test_linear_solver_exact_on_linear_poses(rig):
    idx = landmark_indices()
    w = np.zeros((1, K))
    w[0, SHAPES.index("browDown_L")] = 0.6
    X = features(rig.forward_linear(w)[:, idx], rig.neutral[idx])
    assert np.allclose(LinearSolver(rig, idx).predict(X), w, atol=1e-6)


def test_usd_export_structure(rig, tmp_path):
    w = np.random.default_rng(0).random((5, K))
    info = write_usda(tmp_path / "t.usda", rig, w)
    s = (tmp_path / "t.usda").read_text()
    assert s.startswith("#usda 1.0")
    assert len(re.findall(r'def BlendShape "', s)) == K
    assert len(re.findall(r"^\s+\d+: \[", s, re.M)) == 5  # one time sample per frame
    assert s.count("{") == s.count("}") and s.count("(") == s.count(")")
    assert info["sparse_offsets"] < info["dense_offsets"]
