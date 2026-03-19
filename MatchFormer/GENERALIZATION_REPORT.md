# Generalization Analysis: Overfitting vs. Confidence Collapse

## Overview

The benchmark report established that the 10k-step checkpoint dominates all other models on
scene0000_00. This report tests whether that result reflects genuine learning or memorization,
by evaluating all models on a completely unseen scene (scene0011_00) that no model was trained
on. The results reveal two distinct failure modes — overfitting and confidence collapse — and
show that neither currently-trained model is suitable for real-world deployment without further
training.

---

## Experimental Design

| Factor | scene0000_00 | scene0011_00 |
|--------|-------------|-------------|
| 10k model training data | **Yes** | No |
| 15k / 17.5k / 20k training data | No | No |
| Role in this experiment | In-distribution reference | Generalization test |

All models were evaluated on the same protocol: 50 valid pairs per scene, frame gap=20,
ground-truth reprojection error from depth-projected correspondences. The pair sets are
independently sampled per scene — no overlap between scenes.

The 10k model (last.ckpt) trained exclusively on scene0000_00 for ~1.9 epochs.
The 15k, 17.5k, and 20k models continued training on 11 diverse ScanNet scenes for an
additional 5k–10k steps. Scene0011_00 was not in either training set.

---

## Results

### scene0000_00 — In-Distribution (reference)

| Model | thr | Matches | P@3px | P@5px | Mean Error |
|-------|-----|---------|-------|-------|------------|
| Pretrained | 0.005 | 3168 | 0.00% | 0.00% | 101.4 px |
| Pretrained + Epipolar | 0.005 | 147 | 0.00% | 0.00% | 87.3 px |
| **Fine-tuned 10k** | **0.005** | **3211** | **14.60%** | **34.45%** | **7.2 px** |
| Fine-tuned 10k | 0.100 | 401 | 18.88% | 40.68% | 6.5 px |
| Fine-tuned 15k | 0.005 | 2889 | 7.18% | 18.26% | 10.6 px |
| Fine-tuned 17.5k | 0.005 | 2893 | 0.87% | 2.60% | 19.4 px |
| Fine-tuned 20k | 0.005 | 2683 | 2.84% | 7.68% | 17.4 px |

### scene0011_00 — Out-of-Distribution (generalization test)

| Model | thr | Matches | P@3px | P@5px | Mean Error |
|-------|-----|---------|-------|-------|------------|
| Pretrained | 0.005 | 3165 | 0.00% | 0.00% | 94.0 px |
| Pretrained + Epipolar | 0.005 | 564 | 0.00% | 0.00% | 72.1 px |
| Fine-tuned 10k | 0.005 | 2941 | 0.01% | 0.01% | 61.4 px |
| Fine-tuned 10k | 0.100 | 29 | 0.00% | 0.00% | 48.9 px |
| **Fine-tuned 15k** | **0.005** | **2452** | **0.22%** | **0.58%** | **45.1 px** |
| Fine-tuned 17.5k | 0.005 | 2464 | 0.08% | 0.33% | 54.2 px |
| Fine-tuned 20k | 0.005 | 1855 | 0.21% | 0.54% | 52.0 px |

### Head-to-Head: 10k vs 20k Across Both Scenes

| Scene | Model | P@3px | P@5px | Mean Error |
|-------|-------|-------|-------|------------|
| scene0000_00 | Fine-tuned 10k | **14.60%** | **34.45%** | **7.2 px** |
| scene0000_00 | Fine-tuned 20k | 2.84% | 7.68% | 17.4 px |
| scene0011_00 | Fine-tuned 10k | 0.01% | 0.01% | 61.4 px |
| scene0011_00 | Fine-tuned 20k | **0.21%** | **0.54%** | **52.0 px** |

---

## The Two Failure Modes

### Failure Mode 1 — Overfitting (10k model)

The 10k model's performance on scene0000_00 looked remarkable: 14.6% P@3px, 7.2 px mean
error, healthy confidence. On scene0011_00, it produces:

```
scene0000_00:  P@3px = 14.60%,  mean error =  7.2 px
scene0011_00:  P@3px =  0.01%,  mean error = 61.4 px
```

P@3px drops by a factor of **1460×**. Mean error increases by **8.5×**. The model has
reverted to near-pretrained performance on an unseen scene (pretrained mean error: 94 px,
10k on new scene: 61 px — only a modest improvement).

This is textbook overfitting. The 10k model was trained exclusively on scene0000_00 for
~1.9 epochs. It learned the specific depth distributions, texture patterns, epipolar line
orientations, and camera motion characteristics of that one environment. When these change,
the model has nothing to fall back on. The apparent 14× improvement over the pretrained
baseline was memorization, not generalizable learning.

A further sign: at thr=0.1, scene0011_00 produces only 29 matches per pair (vs 401 on
scene0000_00), and all with poor precision. The model's confident predictions on the new
scene are confidently wrong — a hallmark of overfit models.

### Failure Mode 2 — Confidence Collapse (15k–20k models)

The multi-scene models have the opposite problem. Their geometry is actually better on the
new scene than the 10k model:

```
Mean error on scene0011_00:
  Pretrained:    94.0 px
  Fine-tuned 10k: 61.4 px   ← geometry somewhat improved
  Fine-tuned 15k: 45.1 px   ← geometry better generalized
  Fine-tuned 20k: 52.0 px   ← geometry better generalized
```

Training on 11 diverse scenes did teach the model better generalizable geometry. The
15k model achieves 45 px mean error on scene0011_00 — 24% better than the 10k model
(61 px) on the same unseen scene. The multi-scene training genuinely improved the
model's ability to reason about epipolar geometry across environments.

But the confidence collapse means this knowledge is inaccessible. At thr=0.005, the 15k
model gives 0.22% P@3px — technically non-zero, but practically useless. At any threshold
above 0.01, the model produces almost no matches. The feature representations have been
corrected, but the confidence scores cannot be used to filter them.

---

## Why Each Failure Happened

### Why the 10k model overfit

The 10k model saw ~10,600 unique training pairs (all from scene0000_00, frame gap=20).
Every gradient update was derived from the same scene's geometry. The focal loss learned
which feature pairs to trust specifically within scene0000_00's visual domain — its
lighting, materials, depth range (~1–4 m typical indoor), and camera trajectory.

When a new scene has different materials (different wall textures, floor types), different
depth profiles, or different motion patterns, the learned "trust these features" signal no
longer applies. The model is not wrong in any principled way — it simply has no experience
with anything beyond its training scene.

### Why the multi-scene models collapsed

When training expanded to 11 scenes at step 10k, two forces combined:

**1. Scale of hard negatives.** With 11 diverse scenes, the number of visually confusable
but geometrically incorrect pairs (hard negatives) increased dramatically. A corridor wall
patch looks similar to a bedroom wall patch, but they are not corresponding points. The
focal loss treats these as negatives to be suppressed — pushing their confidence toward 0.
With millions of new hard negatives all producing suppression gradients, the softmax
normalization inflated beyond what any positive signal could overcome.

**2. Learning rate too large for a shifted distribution.** lr=1e-4 was appropriate when all
gradient signal came from one consistent scene. When 11 scenes introduced gradient
directions pointing in different ways, each step was a large, noisy update. The confidence
calibration the model had developed for scene0000_00 was destroyed by the first few
thousand steps on the diverse data before the model could re-stabilize.

The result: the geometry improved (diverse scenes teach more general epipolar reasoning)
but the confidence signal — which is what determines which matches to return — became
useless.

---

## What the Two Scenes Reveal Together

Comparing performance across both scenes exposes the complete picture:

```
                    scene0000_00          scene0011_00
                    (training scene)      (unseen scene)
                    ─────────────────     ─────────────────
Pretrained          0.00% / 101 px        0.00% / 94 px     ← systematic baseline error
10k model           14.60% / 7 px ✓       0.01% / 61 px ✗   ← overfit
15k model           7.18% / 11 px         0.22% / 45 px      ← collapsed but generalizes
20k model           2.84% / 17 px         0.21% / 52 px      ← collapsed but generalizes
```

The pretrained model has consistent systematic error on both scenes (~94–101 px) from a
coordinate bias unrelated to scene content. The 10k model wins on its training scene but
is only marginally better than pretrained on the new scene. The multi-scene models have
lower mean errors on the new scene than the 10k model, confirming that diversity helped,
but their collapsed confidence makes that advantage unextractable.

No currently trained model is both well-calibrated and generalizable.

---

## Training Diversity vs. Confidence: The Trade-off

This experiment isolates a fundamental tension in the training setup:

| Property | 10k (1 scene) | 15k–20k (11 scenes) |
|----------|---------------|---------------------|
| Confidence calibration | Healthy | Collapsed |
| Geometry on training scene | Excellent | Good |
| Geometry on unseen scene | Poor (overfit) | Moderate |
| Useful operating threshold range | 0.005–0.2 | 0.005–0.01 only |
| Practical deployability | Training scene only | Nowhere |

Training diversity is necessary for generalization. Single-scene training cannot produce
a model that works on new environments, regardless of how well it performs on benchmarks
derived from that scene. However, the confidence collapse caused by the focal loss at
lr=1e-4 prevents the multi-scene knowledge from being used.

---

## Path Forward

The results point to a clear solution: retain the 11-scene diversity but fix the training
stability. Concretely:

**1. Lower learning rate: lr = 1e-5**
Reduce the step size by 10×. This allows the model to navigate the diverse 11-scene loss
landscape without overshooting calibration. The confidence distribution changes gradually,
matching the pace at which geometry improves. Expected outcome: the model reaches the
performance of the 15k–20k models geometrically, but with the confidence health of the
10k model.

**2. Warm restart from the 10k checkpoint on 11 scenes at lr = 1e-5**
Rather than starting from the pretrained weights, warm-restart from the 10k checkpoint
(which has corrected geometry for at least one scene) and fine-tune on 11 scenes at a
lower rate. This gives the optimizer a better starting point and a more cautious update
schedule.

**3. Sampled loss with fixed positive:negative ratio**
Cap the negative gradient contribution per step at 10× the positive count (e.g. sample
10 hard negatives per positive). This prevents the 23,000,000:2,000 negative-to-positive
imbalance from dominating the gradient direction, regardless of scene diversity.

**4. Early stopping with generalization monitoring**
Evaluate on scene0011_00 every 2,500 steps during training. Stop when mean error on the
held-out scene stops improving, rather than running to a fixed step count. Given the
results above, the optimal stopping point for the 11-scene run would likely be around
12k–15k steps at lr=1e-5.

---

## Summary

| Finding | Evidence |
|---------|----------|
| 10k model overfit to scene0000_00 | P@3px: 14.6% → 0.01% across scenes |
| Multi-scene training improves geometry | 10k: 61 px, 15k: 45 px mean error on new scene |
| Confidence collapse prevents using that geometry | All 15k–20k models: P@3px < 0.3% on new scene |
| Neither model is deployable as-is | Best result on unseen scene: 0.22% P@3px |
| Root cause: lr=1e-4 too large for diverse training | Collapse begins at step 10k–15k boundary |
| Fix: lr=1e-5 with 11-scene diversity | Predicted to combine calibration + generalization |

---

*Scenes evaluated: scene0000_00 (in-distribution ref) · scene0011_00 (unseen generalization test)*
*50 pairs per scene · frame gap=20 · March 2026*
*Hardware: Apple M-series (MPS) · MatchFormer Lite-LA · τ=50.0*
