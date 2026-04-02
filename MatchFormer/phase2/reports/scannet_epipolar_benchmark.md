# ScanNet Benchmark: Vanilla vs Vanilla + Gaussian Epipolar Masking

**Date:** 2026-04-02
**Model:** Pretrained `indoor-lite-LA.ckpt` (litela, D_MODEL=192, indoor)
**Scene:** scene0000_00
**Split:** 90/10 train/val (seed=42), 555 val pairs
**Frame gap:** 20
**Confidence threshold:** 0.2
**Epipolar τ:** 10.0 (Gaussian decay: `exp(-d² / (2τ²))`)

---

## Method

The GT reprojection distance is computed per match as follows:

1. For each predicted match `(x0, y0)` in image 0, look up the depth `z` from the depth map.
2. Unproject to 3D in camera 0: `P_cam0 = z * K⁻¹ @ [x0, y0, 1]`.
3. Transform to camera 1: `P_cam1 = T_0to1 @ P_cam0`.
4. Project back to image 1: `(x1_gt, y1_gt) = K @ P_cam1 / z1`.
5. Error = Euclidean distance between predicted `(x1_pred, y1_pred)` and GT `(x1_gt, y1_gt)`.

Matches with invalid depth (≤0.1m or >10.0m) or negative reprojected depth are excluded.

**Gaussian epipolar masking** multiplies the dual-softmax confidence matrix by `exp(-d² / (2τ²))`, where `d` is the pixel distance from each image 1 coarse cell to the epipolar line of the corresponding image 0 cell. The fundamental matrix F is computed from GT poses and intrinsics.

---

## Results

| Metric | Vanilla | Vanilla + Gaussian Epipolar (τ=10) | Δ |
|--------|---------|-------------------------------------|---|
| **Mean error** | 2.24 px | **2.19 px** | −0.05 |
| **Median error** | 1.78 px | **1.76 px** | −0.02 |
| <1px | 22.8% | **23.1%** | +0.3% |
| <3px | 77.8% | **78.4%** | +0.6% |
| <5px | 93.8% | **94.2%** | +0.4% |
| <10px | 99.3% | **99.4%** | +0.1% |

---

## Analysis

The improvement from epipolar masking on ScanNet scene0000 is **marginal** (~0.05px mean, ~0.5% precision). This is expected for two reasons:

1. **Short baseline.** ScanNet pairs at frame gap 20 have small viewpoint changes. The vanilla model already achieves 1.78px median error — there are very few gross mismatches for the epipolar constraint to correct.

2. **In-distribution data.** The indoor-lite-LA model was pretrained on ScanNet. It performs near its ceiling on this data. The epipolar mask primarily helps on out-of-distribution or wide-baseline scenes where the model makes larger errors.

### Comparison with WxBS Results

For context, the same Gaussian epipolar masking (τ=10) on WxBS scenes using the outdoor-large-LA model:

| Scene | Vanilla <10px | + Gaussian <10px | Δ |
|-------|---------------|------------------|---|
| dh | 65.7% | **79.3%** | **+13.6%** |
| strahov | 10.0% | **40.0%** | **+30.0%** |
| ministry | 25.0% | **26.9%** | +1.9% |
| kremlin | 27.4% | 27.8% | +0.4% |
| scene0000 | 99.3% | 99.4% | +0.1% |

**Takeaway:** Epipolar masking provides the greatest benefit on hard, wide-baseline scenes where the model makes frequent gross errors. On easy in-distribution scenes, the model is already accurate and the constraint adds negligible value.

---

## Notes

- F matrix computed from GT poses (oracle setting). In practice, F would need to come from IMU, prior frames, or another estimator.
- The `run_benchmark.py` script uses Laplacian decay (`exp(-d/τ)`) with τ=50 by default. This benchmark used Gaussian decay (`exp(-d²/(2τ²))`) with τ=10 via a separate script.
- 555 val pairs evaluated on CPU (~6 min per variant).
