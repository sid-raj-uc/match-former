# Vanilla MatchFormer Lite-LA: Easy vs Hard Scenes

**Date:** 2026-04-01
**Model:** Pretrained `indoor-lite-LA.ckpt` (no fine-tuning)

This report compares vanilla MatchFormer performance on its native domain (ScanNet scene0000 — short-baseline indoor) against a challenging out-of-distribution benchmark (WxBS v1.1 — wide-baseline, appearance/sensor changes).

---

## Datasets

| | ScanNet scene0000 | WxBS v1.1 |
|---|---|---|
| **Type** | Indoor RGB-D, short baseline | Outdoor/mixed, wide baseline |
| **Pairs** | 555 val pairs (10% split, seed=42) | 34 image pairs |
| **Baseline** | ~9cm translation, ~11° rotation | Up to ~90°+ viewpoint change |
| **Challenges** | None (in-domain) | Appearance, illumination, sensor, large viewpoint |
| **GT** | Depth + pose → reprojection | Manually annotated correspondences |
| **Metric** | Pose AUC + reprojection distance | Symmetric epipolar distance on GT pts |

---

## ScanNet scene0000 — In-Domain Results

| Threshold | AUC@5° | AUC@10° | AUC@20° | Precision | Matches | Mean err | Med err | P@3px | P@5px |
|-----------|--------|---------|---------|-----------|---------|----------|---------|-------|-------|
| thr=0.2   | 23.03  | 43.65   | 64.54   | 64.8%     | 2,879   | 2.21 px  | 1.77 px | 77.9% | 93.9% |
| thr=0.05  | 21.12  | 42.84   | 64.89   | 62.8%     | 3,173   | 2.41 px  | 1.83 px | 75.9% | 92.4% |
| thr=0.01  | 21.98  | 42.99   | 65.08   | 62.2%     | 3,217   | 2.45 px  | 1.83 px | 75.7% | 92.2% |
| thr=0.2 + Epipolar | 22.81 | 43.97 | 65.83 | 66.1% | 2,604 | 2.11 px | 1.72 px | 79.9% | 95.1% |

**Best config: thr=0.2** — highest precision, best reprojection accuracy. Lower thresholds add matches but noise increases.

---

## WxBS v1.1 — Out-of-Distribution Results (Indoor-Lite-LA)

### Overall (34 pairs, 233 GT points)

| Metric | Value |
|--------|-------|
| Pairs with valid F estimation | 6 / 34 (18%) |
| Median epipolar error | 8.53 px |
| Mean epipolar error | 71.39 px |
| % GT @ 1px | 14.2% |
| % GT @ 3px | 33.5% |
| % GT @ 5px | 41.2% |
| Avg predicted matches / pair | 19.9 |
| Avg RANSAC inliers / pair | 11.9 |

### By Category

| Category | Description | Valid F | Median epi | @3px | @5px | Avg pred |
|----------|-------------|---------|-----------|------|------|----------|
| WLABS | Illumination + Appearance, mild viewpoint | 1/4 | 1.60 px | 62.7% | 72.5% | 89.0 |
| WGABS | Geometric + Appearance | 1/5 | 8.47 px | 42.1% | 47.4% | 28.0 |
| WGALBS | Geometric + Appearance + Illumination | 4/10 | 11.95 px | 24.3% | 33.9% | 11.9 |
| WGLBS | Large baseline stereo | 1/9 | 103.31 px | 6.9% | 6.9% | 6.1 |
| WGBS | Geometric (viewpoint) only | 0/1 | N/A | N/A | N/A | 7.0 |
| WGSBS | Geometric + Sensor (thermal) | 0/5 | N/A | N/A | N/A | 0.2 |

---

## Head-to-Head Comparison

| Dimension | ScanNet (Easy) | WxBS (Hard) |
|-----------|---------------|-------------|
| Match success rate | ~100% of pairs | 18% of pairs |
| Avg matches / pair | ~5.2 per val pair | 19.9 (but 82% fail) |
| Reprojection / epipolar @ 3px | 77.9% | 33.5% |
| Pose AUC@20° | 64.5% | N/A (no GT poses) |
| Thermal / cross-modal | N/A | 0% success (0 matches) |
| Large viewpoint (>45°) | Rarely seen | Dominant failure mode |

---

## Analysis

### Why ScanNet is easy for this model

The model was pretrained on ScanNet — short-baseline indoor RGB pairs with consistent lighting, dense textures, and small viewpoint changes. On scene0000, it produces confident, high-precision matches on nearly every pair. The reprojection error of 1.77px median shows the matches are geometrically accurate.

### Why WxBS is hard

**Match sparsity.** 24 of 34 pairs produce fewer than 8 matches — the model abstains rather than hallucinating, which is correct behavior but means F estimation is impossible. This is a consequence of the conservative default threshold (0.2) and the large domain gap.

**Domain gap breakdown by challenge type:**
- *Illumination/appearance only (WLABS):* Partially handled — 1 pair succeeds well (kpi: 1.60px median). The mild viewpoint change is within training distribution.
- *Geometric + appearance (WGABS/WGALBS):* Mostly fails. Outdoor scenes with summer/winter, night/day, or large viewpoint differences overwhelm the pretrained features.
- *Large baseline (WGLBS):* Nearly complete failure. The model was never trained on pairs with >20° rotation; the attention patterns learned for local matching don't generalize.
- *Thermal sensor (WGSBS):* Complete failure. The model has no concept of thermal imagery — the appearance domain is entirely foreign.

### The kremlin case

A striking example of the domain gap: kremlin 01.png is a summer street-level photo of St. Basil's Cathedral; 02.png is an aerial winter shot with heavy snow. The model produces **0 matches**. With the outdoor-large-LA model (trained on outdoor data), this pair produces 409 matches with a median epipolar error of 2.15px — showing that the gap is in training data, not architecture.

---

## Outdoor-Large-LA on WxBS (for comparison)

Swapping in the outdoor-pretrained model dramatically changes results:

| Metric | Indoor-Lite | Outdoor-Large |
|--------|------------|---------------|
| Valid pairs | 6 / 34 | **33 / 34** |
| Avg pred matches | 19.9 | **252.3** |
| Median epi error | 8.53 px | **4.22 px** |
| @ 3px | 33.5% | **42.9%** |
| @ 5px | 41.2% | **52.9%** |
| Pose AUC@5° | — | **5.5%** |
| Pose AUC@10° | — | **12.8%** |
| Pose AUC@20° | — | **23.2%** |

The outdoor model closes much of the gap but still struggles with WGSBS (thermal) and the hardest large-baseline pairs.

---

## Summary

Vanilla MatchFormer Lite-LA is a strong indoor short-baseline matcher (Pose AUC@20° = 64.5%, P@3px = 77.9% on ScanNet), but generalizes poorly to the wide-baseline, appearance-shifted, and cross-modal pairs in WxBS. Only 18% of WxBS pairs produce enough matches for geometry estimation. The outdoor-large model recovers significantly (33/34 pairs, better epipolar accuracy), demonstrating that the bottleneck is training distribution rather than architecture. WxBS serves as a useful stress-test to quantify how far these models are from true wide-baseline generalization.
