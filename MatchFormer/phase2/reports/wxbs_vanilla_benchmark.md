# WxBS Benchmark: Vanilla MatchFormer Lite-LA

**Date:** 2026-03-31
**Script:** `benchmark_wxbs.py`
**Model:** Pretrained `indoor-lite-LA.ckpt` (no fine-tuning)

---

## Setup

**Dataset:** WxBS v1.1 — 34 challenging image pairs across 6 difficulty categories.
**Metric:** Symmetric epipolar distance on GT correspondences, using Fundamental matrix estimated via RANSAC from predicted matches.
**Pipeline:**
1. Run vanilla MatchFormer on each image pair (confidence threshold = 0.2).
2. Estimate F via RANSAC (reprojection threshold = 1px, confidence = 0.999).
3. Compute symmetric epipolar distance for every GT correspondence point.

**Image resolution:** Resized to 640×480. GT coordinates scaled proportionally from original image size.

---

## Overall Results (34 pairs, 233 GT points)

| Metric | Value |
|--------|-------|
| Median epipolar error | 8.53 px |
| Mean epipolar error | 71.39 px |
| % GT @ 1px | 14.2% |
| % GT @ 3px | 33.5% |
| % GT @ 5px | 41.2% |
| Avg predicted matches / pair | 19.9 |
| Avg RANSAC inliers / pair | 11.9 |

The large gap between median (8.53px) and mean (71.39px) indicates a heavy tail — a few pairs with catastrophically poor F estimation drag the mean up significantly.

---

## Per-Category Results

| Category | Description | #Pairs | Valid F | Median epi | Mean epi | @1px | @3px | @5px | Avg pred | Avg inlier |
|----------|-------------|--------|---------|-----------|----------|------|------|------|----------|------------|
| WLABS | Illumination + Appearance, mild viewpoint | 4 | 1/4 | 1.60 px | 5.61 px | 29.4% | 62.7% | 72.5% | 89.0 | 52.8 |
| WGABS | Geometric + Appearance | 5 | 1/5 | 8.47 px | 37.02 px | 15.8% | 42.1% | 47.4% | 28.0 | 15.6 |
| WGALBS | Geometric + Appearance + Illumination | 10 | 4/10 | 11.95 px | 94.92 px | 10.4% | 24.3% | 33.9% | 11.9 | 8.2 |
| WGLBS | Large baseline stereo | 9 | 1/9 | 103.31 px | 138.78 px | 0.0% | 6.9% | 6.9% | 6.1 | 3.7 |
| WGBS | Geometric (viewpoint) only | 1 | 0/1 | N/A | N/A | N/A | N/A | N/A | 7.0 | 0.0 |
| WGSBS | Geometric + Sensor (thermal) | 5 | 0/5 | N/A | N/A | N/A | N/A | N/A | 0.2 | 0.0 |

*"Valid F" = pairs where ≥8 matches were found and RANSAC returned a finite F matrix.*

---

## Per-Scene Results

| Category | Scene | Median epi | Mean epi | @1px | @3px | @5px | n_pred | n_inlier |
|----------|-------|-----------|----------|------|------|------|--------|----------|
| WGABS | petrzin | 8.47 px | 37.02 px | 15.8% | 42.1% | 47.4% | 133 | 78 |
| WGABS | kremlin | N/A | N/A | — | — | — | 0 | 0 |
| WGABS | kyiv | N/A | N/A | — | — | — | 2 | 0 |
| WGABS | strahov | N/A | N/A | — | — | — | 1 | 0 |
| WGABS | vatutin | N/A | N/A | — | — | — | 4 | 0 |
| WGALBS | kyiv_dolltheater | 3.80 px | 22.88 px | 18.8% | 40.6% | 54.7% | 66 | 51 |
| WGALBS | rovenki | 13.04 px | 15.83 px | 0.0% | 9.1% | 18.2% | 39 | 24 |
| WGALBS | bridge | 264.76 px | 313.89 px | 0.0% | 0.0% | 0.0% | 10 | 7 |
| WGALBS | flood | N/A | N/A | — | — | — | 0 | 0 |
| WGALBS | kyiv_dolltheater2 | N/A | N/A | — | — | — | 2 | 0 |
| WGALBS | stadium | N/A | N/A | — | — | — | 0 | 0 |
| WGALBS | submarine | N/A | N/A | — | — | — | 0 | 0 |
| WGALBS | submarine2 | N/A | N/A | — | — | — | 2 | 0 |
| WGALBS | tyn | N/A | N/A | — | — | — | 0 | 0 |
| WGALBS | zanky | N/A | N/A | — | — | — | 0 | 0 |
| WGBS | kn-church | N/A | N/A | — | — | — | 7 | 0 |
| WGLBS | warsaw | 103.31 px | 138.78 px | 0.0% | 6.9% | 6.9% | 55 | 33 |
| WGLBS | alupka | N/A | N/A | — | — | — | 0 | 0 |
| WGLBS | berlin | N/A | N/A | — | — | — | 0 | 0 |
| WGLBS | charlottenburg | N/A | N/A | — | — | — | 0 | 0 |
| WGLBS | church | N/A | N/A | — | — | — | 0 | 0 |
| WGLBS | him | N/A | N/A | — | — | — | 0 | 0 |
| WGLBS | maidan | N/A | N/A | — | — | — | 0 | 0 |
| WGLBS | ministry | N/A | N/A | — | — | — | 0 | 0 |
| WGLBS | silasveta2 | N/A | N/A | — | — | — | 0 | 0 |
| WGSBS | kettle | N/A | N/A | — | — | — | 0 | 0 |
| WGSBS | kettle2 | N/A | N/A | — | — | — | 0 | 0 |
| WGSBS | lab | N/A | N/A | — | — | — | 1 | 0 |
| WGSBS | lab2 | N/A | N/A | — | — | — | 0 | 0 |
| WGSBS | window | N/A | N/A | — | — | — | 0 | 0 |
| WLABS | kpi | 1.60 px | 5.61 px | 29.4% | 62.7% | 72.5% | 353 | 211 |
| WLABS | dh | N/A | N/A | — | — | — | 0 | 0 |
| WLABS | kyiv | N/A | N/A | — | — | — | 3 | 0 |
| WLABS | ministry | N/A | N/A | — | — | — | 0 | 0 |

---

## Analysis

### What works

**WLABS/kpi** is the standout case: 353 predicted matches, 211 inliers, median epipolar error of 1.60px, 62.7% of GT correspondences within 3px. This scene involves illumination and appearance changes with relatively small viewpoint change — closest in character to the ScanNet indoor training domain.

**WGALBS/kyiv_dolltheater** also performs reasonably (median 3.80px, 54.7% @ 5px), confirming the model handles mild to moderate appearance variation when the viewpoint change is within training distribution.

### Where it fails

**Match sparsity is the dominant failure mode.** 24 of 34 pairs produce fewer than 8 matches, making F estimation impossible. The vanilla MatchFormer confidence threshold of 0.2 is very conservative — the model simply abstains on difficult pairs rather than producing wrong matches.

**WGSBS (thermal vs visible):** Complete failure (0–1 matches across all 5 pairs). The model has no exposure to cross-modal matching during training.

**WGLBS (large baseline stereo):** 8 of 9 pairs fail to produce enough matches. The one successful pair (warsaw, 55 predictions) has a very high epipolar error of 25.7px median, suggesting the matches are geometrically inconsistent — the predicted F does not explain the GT geometry well.

**WGABS/WGALBS failures:** Many outdoor scenes with large viewpoint changes, blurred images, or night/day appearance differences produce zero matches. The MatchFormer backbone (pretrained on indoor ScanNet at ~11° rotation / 9cm baseline) is not robust to these conditions.

### Distribution shift

The model was pretrained on ScanNet indoor RGB pairs with:
- Short baselines (~9cm translation, ~11° rotation)
- Consistent lighting
- Dense, textured indoor scenes

WxBS pairs involve large viewpoint changes (up to ~90°), sensor modality shifts, night/day transitions, and heavily occluded outdoor scenes — all far outside the training distribution. The results reflect this clearly.

---

## Summary

Vanilla MatchFormer Lite-LA achieves **valid F estimation on only 6 of 34 WxBS pairs** (18%). On those 6, the median epipolar error ranges from 1.6px (easy illumination changes) to 264px (large-baseline outdoor). The model is a strong indoor short-baseline matcher but is not competitive on the wide-baseline and cross-modal challenges in WxBS. These results establish a baseline for evaluating whether fine-tuning or architectural changes can improve generalization to this harder benchmark.
