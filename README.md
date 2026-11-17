# facesolve: neural blendshape solver for facial performance capture

A small, end-to-end version of a character-pipeline ML tool: **noisy facial
landmark capture in, clean blendshape animation out, delivered as UsdSkel for
Maya/Houdini.**

```
raw tracker output ──► clean ──► MLP solver ──► blendshape weights ──► .usda (UsdSkel) ──► Maya / Houdini
 (head motion, noise,   (outliers,   (landmarks →      (16 ARKit-style      (BlendShape prims +
  gaps, marker swaps)    gaps, Kabsch  weights in [0,1])  shapes)              SkelAnimation)
                         stabilization)
```

## Why

Hand-keying facial animation doesn't scale, and a linear least-squares solve
can't match a production rig: real rigs have rotational jaws and combination
correctives (smile + jaw open, blink + smile) that sit outside the linear basis.
A learned solver can absorb that nonlinearity and runs fast enough for real time.

## What's in it

| Piece | File | Notes |
|---|---|---|
| Procedural face rig | `facesolve/rig.py` | 1,600-vertex mesh, 16 ARKit-named blendshapes, a true rotational jaw, 5 FACS combination correctives, 72 landmarks (iBUG-68 layout + 4 helmet head markers) |
| Data | `facesolve/capture.py` | **raw capture**: 120-frame takes seen through a moving head (±15° yaw), 0.4 mm noise, occlusion gaps, 15–30 mm marker swaps. **artist set**: curated single-shape and pair poses |
| Cleaning | `facesolve/clean.py` | temporal-median outlier rejection, gap interpolation, Kabsch (SVD) rigid stabilization on the head markers |
| Solver | `facesolve/model.py` | 2×256 ReLU MLP, sigmoid output, Adam + cosine LR, input-noise augmentation; NumPy with hand-derived backprop (gradient-checked). Baseline: bounded least squares (BVLS) |
| PyTorch port | `scripts/train_torch.py` | same network in PyTorch + ONNX export |
| USD export | `facesolve/usd_export.py` | UsdSkel: SkelRoot / Skeleton / SkelAnimation `blendShapeWeights` time samples / Mesh with sparse `BlendShape` offsets |
| Maya | `scripts/maya_apply_weights.py` | keys solved weights onto a `blendShape` node with `maya.cmds` |
| Ops | `Dockerfile`, `.github/workflows/ci.yml` | tests run inside the image build; CI runs tests, a smoke experiment and the Docker build |

## Results (`python scripts/run_experiment.py`, 2 CPUs, about 40 s)

24,000 training frames (cleaned capture + 3× artist set), 4,800 held-out test frames.
Vertex error is measured on all 1,600 vertices of the *nonlinear* rig.

| Solver (on cleaned input) | Weight MAE | Mean vertex err | p95 | Max |
|---|---|---|---|---|
| Bounded least squares | 0.094 | 0.40 mm | 1.44 mm | 10.5 mm |
| **MLP** | **0.060** | **0.29 mm** | **1.03 mm** | **5.9 mm** |
| MLP on *uncleaned* input | 0.083 | 0.45 mm | 1.73 mm | 16.9 mm |

**Cleaning** (test set): landmark RMSE 18.8 mm raw → 2.67 mm after stabilization
alone → **1.26 mm** with outlier rejection; 20,967 missing samples filled; injected
marker swaps caught at 100% recall and precision (synthetic outliers are easy to
spot; real ones won't be).

**Inference**: 282k frames/s batched in float32 (2.3× float64), 58 µs per single
frame, about **70× faster** than the least-squares solve (3.9k frames/s).

**USD**: 120-frame solved take, 16 BlendShape prims, sparse offsets store 9,106 of
25,600 vertex deltas (−64%).

### What doesn't work yet (honest)

* **Artist intent on clean poses.** On held-out curated pair poses the MLP is
  worse than least squares (0.17 mm vs 0.017 mm): it learned the capture
  distribution and drifts on clean, exaggerated poses. Next step: an MLP initial
  guess refined by a few Gauss–Newton steps on the rig, or a vertex-space loss.
* `mouthFunnel` and `mouthPucker` share a region and are the least separable
  shapes (MAE 0.135 / 0.088).
* Everything is synthetic. The `.usda` is written as plain text and should be
  checked with `usdchecker` and opened in usdview / Maya before it's trusted.

## Run

```bash
pip install -r requirements.txt
python -m pytest -q tests            # 11 tests
python scripts/run_experiment.py     # writes results/
pip install torch && python scripts/train_torch.py
docker build -t facesolve . && docker run --rm -v "$PWD/results:/app/results" facesolve
```
