"""PyTorch version of the MLP solver (same data, same architecture, same loss).

    pip install torch
    python scripts/train_torch.py            # writes results/torch_results.json
                                             # and results/mlp_solver.onnx

Compare its numbers with results/results.json (the NumPy reference). They
should agree to within run-to-run noise; if not, one of the two has a bug.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from facesolve.capture import artist_set, simulate_capture  # noqa: E402
from facesolve.model import features  # noqa: E402
from facesolve.rig import K, FaceRig, landmark_indices  # noqa: E402
from run_experiment import clean_all, vertex_error  # noqa: E402

torch.manual_seed(0)
rig = FaceRig()
idx = landmark_indices()
nl = rig.neutral[idx]

tr, va, te = (simulate_capture(rig, n, seed=s) for n, s in ((200, 1), (20, 2), (40, 3)))
W_art, L_art = artist_set(rig)
Xtr = np.concatenate([features(clean_all(tr, nl)[0], nl)] + [features(L_art, nl)] * 3)
Ytr = np.concatenate([tr.weights.reshape(-1, K)] + [W_art] * 3)
Xva, Yva = features(clean_all(va, nl)[0], nl), va.weights.reshape(-1, K)
Xte, Yte = features(clean_all(te, nl)[0], nl), te.weights.reshape(-1, K)

mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6


def T(a):
    return torch.tensor((a - mu) / sd, dtype=torch.float32)


net = nn.Sequential(
    nn.Linear(Xtr.shape[1], 256), nn.ReLU(),
    nn.Linear(256, 256), nn.ReLU(),
    nn.Linear(256, K), nn.Sigmoid(),
)
epochs = 40
opt = torch.optim.Adam(net.parameters(), lr=1e-3)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
loader = torch.utils.data.DataLoader(
    torch.utils.data.TensorDataset(T(Xtr), torch.tensor(Ytr, dtype=torch.float32)),
    batch_size=256, shuffle=True)

t0 = time.perf_counter()
for ep in range(epochs):
    net.train()
    for xb, yb in loader:
        xb = xb + 0.05 * torch.randn_like(xb) / torch.tensor(sd, dtype=torch.float32)
        loss = nn.functional.mse_loss(net(xb), yb)
        opt.zero_grad()
        loss.backward()
        opt.step()
    sched.step()
    if ep % 5 == 4:
        with torch.no_grad():
            v = nn.functional.mse_loss(net(T(Xva)), torch.tensor(Yva, dtype=torch.float32))
        print(f"epoch {ep + 1} val_mse {v.item():.5f}")
train_s = time.perf_counter() - t0

net.eval()
with torch.no_grad():
    P = net(T(Xte)).numpy().astype(np.float64)
ve = vertex_error(rig, Yte, P)
res = {"train_seconds": train_s, "weight_mae": float(np.abs(P - Yte).mean()),
       "mean_mm": ve["mean_mm"], "p95_mm": ve["p95_mm"]}

# export for deployment (ONNX Runtime, or Maya/Houdini plugins that host it)
torch.onnx.export(net, T(Xte[:1]), str(ROOT / "results" / "mlp_solver.onnx"),
                  input_names=["landmark_offsets"], output_names=["blendshape_weights"],
                  dynamic_axes={"landmark_offsets": {0: "frames"}})
json.dump(res, open(ROOT / "results" / "torch_results.json", "w"), indent=1)
print(json.dumps(res, indent=1))
