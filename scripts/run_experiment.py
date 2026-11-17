"""End-to-end experiment: simulate capture -> clean -> train -> evaluate -> export.

    python scripts/run_experiment.py            # full run (about a minute on 2 CPUs)
    python scripts/run_experiment.py --quick    # CI smoke run
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from facesolve.capture import artist_set, simulate_capture  # noqa: E402
from facesolve.clean import clean_sequence, fill_gaps, stabilize  # noqa: E402
from facesolve.model import LinearSolver, MLPSolver, features  # noqa: E402
from facesolve.rig import K, SHAPES, FaceRig, landmark_indices  # noqa: E402
from facesolve.usd_export import write_usda  # noqa: E402


def clean_all(cap, neutral_lm):
    out, masks = [], []
    for raw in cap.raw:
        c, m = clean_sequence(raw, neutral_lm)
        out.append(c), masks.append(m)
    return np.concatenate(out), np.stack(masks)


def uncleaned(cap, neutral_lm):
    """Ablation: gap-fill + stabilize only, no outlier rejection."""
    return np.concatenate([stabilize(fill_gaps(r), neutral_lm) for r in cap.raw])


def vertex_error(rig, w_true, w_pred, chunk=500):
    errs = []
    for s in range(0, len(w_true), chunk):
        a = rig.forward(w_true[s : s + chunk])
        b = rig.forward(w_pred[s : s + chunk])
        errs.append(np.linalg.norm(a - b, axis=-1))
    e = np.concatenate(errs)
    return {"mean_mm": float(e.mean()), "p95_mm": float(np.percentile(e, 95)),
            "max_mm": float(e.max()), "frame_mean_mm": e.mean(1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    n_train, n_val, n_test, epochs = (20, 4, 6, 5) if args.quick else (200, 20, 40, 40)

    out_dir = ROOT / "results"
    out_dir.mkdir(exist_ok=True)
    rig = FaceRig()
    idx = landmark_indices()
    neutral_lm = rig.neutral[idx]
    R = {"config": {"vertices": rig.V, "triangles": len(rig.faces), "landmarks": len(idx),
                    "blendshapes": K, "train_frames": n_train * 120, "test_frames": n_test * 120}}

    # ---------- data ----------
    tr = simulate_capture(rig, n_train, seed=1)
    va = simulate_capture(rig, n_val, seed=2)
    te = simulate_capture(rig, n_test, seed=3)
    W_art, L_art = artist_set(rig)

    # ---------- cleaning ----------
    t0 = time.perf_counter()
    Xtr_lm, _ = clean_all(tr, neutral_lm)
    Xva_lm, _ = clean_all(va, neutral_lm)
    Xte_lm, te_mask = clean_all(te, neutral_lm)
    clean_s = time.perf_counter() - t0
    gt = te.clean.reshape(-1, *te.clean.shape[2:])
    raw_flat = te.raw.reshape(gt.shape)
    ok = ~np.isnan(raw_flat[..., 0])
    tp = (te_mask & te.outlier_mask).sum()
    R["cleaning"] = {
        "raw_rmse_mm": float(np.sqrt(np.mean(np.sum((raw_flat[ok] - gt[ok]) ** 2, -1)))),
        "no_outlier_rejection_rmse_mm": float(np.sqrt(np.mean(np.sum((uncleaned(te, neutral_lm) - gt) ** 2, -1)))),
        "cleaned_rmse_mm": float(np.sqrt(np.mean(np.sum((Xte_lm - gt) ** 2, -1)))),
        "outlier_recall": float(tp / te.outlier_mask.sum()),
        "outlier_precision": float(tp / max(te_mask.sum(), 1)),
        "missing_samples_filled": int((~ok).sum()),
        "frames_per_s": float((len(Xtr_lm) + len(Xva_lm) + len(Xte_lm)) / clean_s),
    }
    print("cleaning", json.dumps(R["cleaning"], indent=1))

    # ---------- training ----------
    Wtr = tr.weights.reshape(-1, K)
    Xtr = np.concatenate([features(Xtr_lm, neutral_lm)] + [features(L_art, neutral_lm)] * 3)
    Ytr = np.concatenate([Wtr] + [W_art] * 3)
    Xva, Yva = features(Xva_lm, neutral_lm), va.weights.reshape(-1, K)
    Xte, Yte = features(Xte_lm, neutral_lm), te.weights.reshape(-1, K)

    mlp = MLPSolver(Xtr.shape[1], K)
    t0 = time.perf_counter()
    hist = mlp.fit(Xtr, Ytr, Xva, Yva, epochs=epochs, noise_std=0.05)
    R["train"] = {"seconds": time.perf_counter() - t0, "epochs": epochs,
                  "params": int(sum(w.size for w in mlp.W) + sum(b.size for b in mlp.b)),
                  "final": hist[-1]}
    mlp.save(out_dir / "mlp_solver.npz")
    json.dump(hist, open(out_dir / "history.json", "w"), indent=1)

    # ---------- evaluation ----------
    lin = LinearSolver(rig, idx)
    preds = {"mlp": mlp.predict(Xte), "linear_lsq": lin.predict(Xte)}
    preds["mlp_uncleaned_input"] = mlp.predict(features(uncleaned(te, neutral_lm), neutral_lm))
    ev = {}
    for k, p in preds.items():
        ve = vertex_error(rig, Yte, p)
        ev[k] = {"weight_mae": float(np.abs(p - Yte).mean()),
                 **{a: b for a, b in ve.items() if a != "frame_mean_mm"}}
    # artist-intent check: held-out curated poses not in the training set
    Wh = []
    for a in range(K):
        for b in range(a + 1, K):
            w = np.zeros(K)
            w[[a, b]] = 0.75
            Wh.append(w)
    Wh = np.asarray(Wh)
    Xh = features(rig.forward(Wh)[:, idx], neutral_lm)
    for k, p in {"mlp": mlp.predict(Xh), "linear_lsq": lin.predict(Xh)}.items():
        ev[k]["artist_holdout_mean_mm"] = vertex_error(rig, Wh, p)["mean_mm"]
        ev[k]["artist_holdout_weight_mae"] = float(np.abs(p - Wh).mean())
    # per-shape error, for the README
    ev["mlp_per_shape_mae"] = dict(zip(SHAPES, np.abs(preds["mlp"] - Yte).mean(0).round(4).tolist()))
    R["eval"] = ev
    print("eval", json.dumps(ev, indent=1))

    # ---------- inference speed ----------
    one = Xte[:1]
    sp = {}
    for name, dt in (("float64", np.float64), ("float32", np.float32)):
        mlp.predict(Xte, dt)
        t0 = time.perf_counter()
        for _ in range(5):
            mlp.predict(Xte, dt)
        batch = (time.perf_counter() - t0) / 5
        t0 = time.perf_counter()
        for _ in range(2000):
            mlp.predict(one, dt)
        single = (time.perf_counter() - t0) / 2000
        sp[name] = {"batch_frames_per_s": len(Xte) / batch, "single_frame_us": single * 1e6}
    t0 = time.perf_counter()
    lin.predict(Xte[:500])
    sp["linear_lsq_frames_per_s"] = 500 / (time.perf_counter() - t0)
    R["inference"] = sp
    print("inference", json.dumps(sp, indent=1))

    # ---------- export one solved take to USD ----------
    take = preds["mlp"][:120]
    R["usd"] = write_usda(out_dir / "solved_take.usda", rig, take)
    np.savetxt(out_dir / "solved_take_weights.csv", take, delimiter=",",
               header=",".join(SHAPES), comments="", fmt="%.4f")

    def clean_json(o):
        if isinstance(o, dict):
            return {k: clean_json(v) for k, v in o.items()}
        if isinstance(o, float):
            return round(o, 4)
        return o

    json.dump(clean_json(R), open(out_dir / "results.json", "w"), indent=1)
    print("wrote", out_dir / "results.json")


if __name__ == "__main__":
    main()
