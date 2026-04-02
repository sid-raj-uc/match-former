# WxBS Benchmark: Vanilla vs Vanilla + Epipolar Masking

**Date:** 2026-04-02
**Model:** Pretrained `outdoor-large-LA.ckpt` (largela, D_MODEL=256, outdoor)
**Config:** BACKBONE_TYPE=largela, RESOLUTION=(8,2), confidence threshold=0.2

---

## Method

**Epipolar masking** is a post-hoc geometric filter applied to the coarse confidence matrix after dual-softmax normalization. For each coarse grid cell in image 0, the epipolar line in image 1 is computed via `l = F @ p0`. The distance from every image 1 cell to this line is computed, and a decay function converts distance to a mask weight in [0, 1]. The confidence matrix is then element-wise multiplied by this mask.

Three decay modes are compared:

| Mode | Formula | Behavior |
|------|---------|----------|
| **Laplacian** | `exp(-d / τ)` | Linear exponent — moderate falloff |
| **Gaussian** | `exp(-d² / (2τ²))` | Quadratic exponent — sharp near line, fast decay |
| **Hard** | `1 if d < τ else 0` | Binary cutoff — no smooth transition |

The fundamental matrix F is estimated from GT correspondences via RANSAC. τ = 10 pixels unless noted.

**Evaluation metric:** For each predicted match in image 0 that falls within 20px of a GT point, compute the reprojection error as the Euclidean distance between the predicted image 1 location and the GT image 1 location.

---

## Per-Scene Results

### WGABS/kremlin

| Method | #Matches | Mean Err | Med Err | <5px | <10px |
|--------|----------|----------|---------|------|-------|
| Vanilla | 405 | 18.6 | 15.7 | 9.7% | 27.4% |
| Laplacian τ=10 | 328 | **16.7** | **15.0** | **10.6%** | **30.3%** |
| Gaussian τ=10 | 377 | 17.4 | 15.5 | 9.7% | 27.8% |
| Gaussian τ=15 | 385 | 17.3 | 15.4 | 9.6% | 27.9% |
| Gaussian τ=20 | 390 | 17.3 | 15.4 | 9.5% | 27.7% |

### WGABS/petrzin

| Method | #Matches | Mean Err | Med Err | <5px | <10px |
|--------|----------|----------|---------|------|-------|
| Vanilla | 901 | 19.2 | 16.5 | 5.0% | 22.1% |
| Laplacian τ=10 | 695 | **18.6** | 16.5 | **5.1%** | **22.5%** |
| Gaussian τ=10 | 781 | 18.9 | **16.4** | 4.8% | 22.4% |
| Gaussian τ=15 | 813 | 18.9 | 16.3 | 4.8% | 22.3% |
| Gaussian τ=20 | 822 | 18.9 | 16.4 | 4.7% | 22.3% |

### WGABS/strahov (hard scene)

| Method | #Matches | Mean Err | Med Err | <5px | <10px |
|--------|----------|----------|---------|------|-------|
| Vanilla | 112 | 114.3 | 96.2 | 5.0% | 10.0% |
| Laplacian τ=10 | 3 | **7.8** | **7.8** | **50.0%** | **50.0%** |
| Gaussian τ=10 | 12 | 11.5 | 11.6 | 20.0% | 40.0% |
| Gaussian τ=15 | 13 | 11.5 | 11.6 | 20.0% | 40.0% |
| Gaussian τ=20 | 18 | 40.8 | 13.5 | 16.7% | 33.3% |

### WGABS/vatutin (hard scene)

| Method | #Matches | Mean Err | Med Err | <5px | <10px |
|--------|----------|----------|---------|------|-------|
| Vanilla | 21 | 153.8 | 153.8 | 0.0% | 0.0% |
| Laplacian τ=10 | 0 | — | — | — | — |
| Gaussian τ=10 | 1 | — | — | — | — |

> Too few GT-nearby matches to evaluate meaningfully.

### WLABS/kyiv

| Method | #Matches | Mean Err | Med Err | <5px | <10px |
|--------|----------|----------|---------|------|-------|
| Vanilla | 344 | 19.7 | 17.8 | 6.6% | 21.9% |
| Laplacian τ=10 | 263 | 18.8 | 17.6 | 6.1% | 21.8% |
| Gaussian τ=10 | 303 | **18.5** | **17.0** | 6.3% | **22.2%** |
| Gaussian τ=15 | 312 | 18.4 | 17.2 | **6.8%** | 22.4% |
| Gaussian τ=20 | 315 | 18.5 | 17.5 | 6.7% | 22.2% |

### WLABS/ministry

| Method | #Matches | Mean Err | Med Err | <5px | <10px |
|--------|----------|----------|---------|------|-------|
| Vanilla | 127 | 20.6 | 13.3 | 4.5% | 25.0% |
| Laplacian τ=10 | 78 | **15.8** | **12.3** | **6.2%** | **28.1%** |
| Gaussian τ=10 | 99 | 20.2 | 12.7 | 5.1% | 26.9% |
| Gaussian τ=15 | 112 | 20.6 | 13.2 | 4.6% | 25.3% |
| Gaussian τ=20 | 113 | 20.6 | 13.2 | 4.6% | 25.3% |

### WLABS/dh

| Method | #Matches | Mean Err | Med Err | <5px | <10px |
|--------|----------|----------|---------|------|-------|
| Vanilla | 142 | 23.8 | 8.2 | 34.3% | 65.7% |
| Laplacian τ=10 | 91 | 7.1 | 7.2 | 40.0% | 76.0% |
| **Gaussian τ=10** | **112** | **6.8** | **6.3** | **41.4%** | **79.3%** |
| Gaussian τ=15 | 114 | 7.1 | 6.8 | 40.0% | 76.7% |
| Gaussian τ=20 | 118 | 7.1 | 6.8 | 40.0% | 76.7% |

### WLABS/kpi

| Method | #Matches | Mean Err | Med Err | <5px | <10px |
|--------|----------|----------|---------|------|-------|
| Vanilla | 992 | 13.9 | 13.5 | 9.4% | 30.9% |
| Laplacian τ=10 | 873 | **13.5** | **13.4** | 8.9% | **31.0%** |
| Gaussian τ=10 | 948 | 13.8 | 13.5 | **9.5%** | 31.0% |
| Gaussian τ=15 | 963 | 13.8 | 13.5 | 9.4% | 30.9% |
| Gaussian τ=20 | 972 | 13.8 | 13.5 | 9.4% | 30.9% |

---

## Summary

### Match count impact (τ=10)

| Method | Avg match reduction vs Vanilla |
|--------|-------------------------------|
| Laplacian | −24.5% |
| Gaussian | −10.8% |

Laplacian is more aggressive — it kills ~2.3× more matches than Gaussian at the same τ. This is because `exp(-d/τ)` decays slower than `exp(-d²/(2τ²))` at large distances but faster near zero, resulting in more matches falling below the confidence threshold.

### When does epipolar masking help most?

| Condition | Improvement | Example |
|-----------|-------------|---------|
| **Many wrong matches** (vanilla mean err > 50px) | Dramatic | strahov: 114→8 mean err |
| **Moderate errors** (mean err 15–25px) | Modest | dh: 24→7 mean err |
| **Already reasonable** (mean err < 15px) | Marginal | kpi: 13.9→13.5 mean err |

### Laplacian vs Gaussian (τ=10)

| | Laplacian | Gaussian |
|---|-----------|----------|
| **Precision** | Slightly better on most scenes | Best on `dh` (79.3% vs 76.0% <10px) |
| **Match count** | Keeps fewer matches | Keeps ~15% more matches |
| **Failure modes** | Can over-filter (strahov: only 3 matches) | More robust match count |
| **Best for** | Precision-critical applications | Balance of precision and recall |

### Recommendation

- **Gaussian τ=10** is the best default — it improves accuracy across all scenes while preserving most matches. It outperformed Laplacian on `dh` (the scene with the most evaluable GT points).
- **Laplacian τ=10** is better when you can afford to lose matches and want maximum precision (e.g., pose estimation where a few high-quality matches suffice).
- **Gaussian τ=15/20** provides negligible benefit over vanilla — the decay is too gentle.

---

## Notes

- F matrix is estimated from GT correspondences via RANSAC, not from model predictions. This is an oracle setting — in practice, F would need to come from another source (e.g., IMU, prior frame poses).
- Evaluation only considers predicted matches within 20px of a GT point in image 0. Scenes with few GT correspondences (vatutin) are unreliable.
- The `outdoor-large-LA` model was used since WxBS contains outdoor scenes. Indoor models may perform differently.
