# MatchFormer Epipolar Fine-Tuning — Project Report

## Overview

Fine-tuned MatchFormer Lite-LA with epipolar geometry supervision on ScanNet indoor scenes.
The goal was to improve geometrically-consistent image matching by incorporating camera pose
information during training, without catastrophic forgetting of the pretrained features.

---

## Architecture

**Model**: MatchFormer Lite-LA — a hierarchical transformer with 4 attention stages.

| Stage | Scale | Cross-Attention | Role |
|-------|-------|----------------|------|
| AttentionBlock1 | 1/4 | No | Low-level edge/texture features |
| AttentionBlock2 | 1/8 | No | Mid-level features |
| AttentionBlock3 | 1/16 | Yes | Coarse cross-image matching |
| AttentionBlock4 | 1/32 | Yes | Final cross-image matching |
| FPN | — | — | Merges stages into coarse (c2_out) and fine (c1_out) features |

**Key insight**: `CoarseMatching` and `FineMatching` modules have no learnable parameters.
All trainable weights live in the backbone. The similarity matrix is computed as:

```
conf[i,j] = softmax(sim, dim=1)[i,j] * softmax(sim, dim=2)[i,j]
```

Dual-softmax enforces mutual agreement — `conf[i,j]` is high only if patch `i` in image 0
thinks `j` is its best match AND patch `j` thinks `i` is its best match.

---

## Training Pipeline

### Data
- **Dataset**: ScanNet indoor scenes (scenes 0000–0010, ~50k+ pairs)
- **Pair construction**: consecutive frames with `frame_gap=20` (~0.67s at 30fps)
- **Input**: 640×480 grayscale images, coarse grid 60×80 = 4800 patches
- **GT matches**: depth reprojection — unproject coarse grid from image 0 using depth + pose,
  project into image 1, snap to nearest coarse cell (~3000 GT matches per pair)

### Training Step (4-pass structure)
1. **Eval pass** (no_grad) — populates spatial dims `hw0_c/hw1_c`
2. **Save `ref_conf_matrix`** — snapshot of current inference distribution (before clearing)
3. **Compute supervision** — GT coarse correspondence indices from depth reprojection
4. **Clear batch keys** — remove stale conf/sim matrices
5. **Train pass** — real forward with gradients
6. **Compute loss**

### Epipolar Injection
Per-batch fundamental matrix `F` computed from camera poses, injected as a soft mask:
```
epipolar_mask[i,j] = exp(-dist(epipolar_line_i, patch_j) / tau)
conf_matrix = conf_matrix * epipolar_mask
```
`tau=50.0` — soft mask, allows some slack to handle multi-scene diversity.

---

## Loss Function

**Final**: `FocalLoss(alpha=0.5, gamma=2.0, neg_per_pos=0)` on `conf_matrix`

```
loss_pos = -alpha * (1-p)^gamma * log(p)        for GT positive cells
loss_neg = -(1-alpha) * p^gamma * log(1-p)      for all negative cells
loss_c   = (loss_pos + loss_neg).sum() / num_positives
loss_f   = L2 on fine-level offset predictions (weighted by inverse std)
total    = 1.0 * loss_c + 0.5 * loss_f
```

### Loss Evolution

| Phase | Loss | Issue |
|-------|------|-------|
| Original | FocalLoss (α=0.5, γ=2.0, neg_per_pos=0) | Confidence collapse on 4800×4800 grid |
| Experiment | TripletCoarseLoss (margin=1.0) on sim_matrix | Zero gradients — pretrained model already satisfies margin |
| + KL reg | Triplet + KL(frozen‖current) | Complex, frozen model overhead |
| **Final** | **FocalLoss (α=0.5, γ=2.0, neg_per_pos=0)** | Reverted — combined with frozen backbone |

---

## Layer Freezing Strategy

**Final strategy**: Freeze all → selectively unfreeze

| Module | Trainable | Reason |
|--------|-----------|--------|
| AttentionBlock1 | No | Low-level self-attention, stable |
| AttentionBlock2 | No | Mid-level self-attention, stable |
| **AttentionBlock3** | **Yes** | Cross-attention, coarse matching |
| **AttentionBlock4** | **Yes** | Cross-attention, final stage |
| layer2–4_outconv | No | Coarse FPN, kept fixed |
| **layer1_outconv + layer1_outconv2** | **Yes** | Fine feature FPN head |

**Trainable params**: 13,144,704 / 20,256,704 = **64.9%**

### Freezing Strategy Evolution

| Phase | Strategy | Result |
|-------|----------|--------|
| Phase 1 | All layers | Unstable, confidence collapse |
| Phase 2 | All layers, neg_per_pos=15 | Flat loss, no learning |
| Phase 3 | All layers, triplet loss | Zero gradients |
| Phase 4 | Freeze backbone entirely | CoarseMatching/FineMatching have no params — nothing to train |
| **Phase 5** | **Freeze all → unfreeze AB3, AB4, fine FPN** | Current |

---

## W&B Metrics Logged

| Metric | Description |
|--------|-------------|
| `train/loss` | Total loss |
| `train/loss_c` | Focal coarse loss |
| `train/loss_f` | Fine L2 loss |
| `train/num_gt_matches` | GT supervision pairs per batch (should be ~3000×batch_size) |
| `train/num_matches` | Predicted matches surviving threshold (fixed at 7680 in train — constant by design) |
| `train/conf_max` | Max confidence value — collapse detector |
| `train/conf_mean` | Mean confidence value |
| `val/loss`, `val/loss_c`, `val/loss_f` | Validation losses |
| `val/num_gt_matches` | Validation GT matches |
| `val/num_matches` | True predicted match count (meaningful — no GT padding in val) |

---

## Benchmark Results

Evaluated on 100 pairs, `tau=50.0`, threshold sweep. Metric: pixel reprojection error on predicted fine matches.

### scene0000_00 — In-Distribution (seen during training)

| Model | Thr | Mean Err | P@3px | P@5px | Avg Matches |
|-------|-----|----------|-------|-------|-------------|
| Baseline (pretrained) | 0.005   | 92.53px | 0.03% | 0.08% | 2723 |
| Run 13 (fine-tuned)   | 0.0001  | 21.42px | 1.68% | 4.53% | 1643 |
| Run 13 (fine-tuned)   | 0.001   | 21.19px | 1.69% | 4.57% | 1570 |
| Run 13 (fine-tuned)   | 0.002   | 20.71px | 1.78% | 4.74% | 1228 |
| **Run 13 (fine-tuned)**   | **0.005**   | **20.47px** | **1.84%** | **4.88%** | **439** |
| Run 13 (fine-tuned)   | 0.01    | 21.92px | 1.28% | 3.88% | 117  |
| Run 13 (fine-tuned)   | 0.02    | 26.47px | 0.52% | 1.92% | 29   |

**Finding**: Fine-tuning improved mean error by ~4.5× (92px → 20px) and P@3px by ~60× (0.03% → 1.84%).
**Optimal threshold is 0.005** — going lower adds noisy matches that increase error without improving precision.
The baseline outputs many (2723) low-quality matches; the fine-tuned model outputs fewer (439) but more accurate matches.

### scene0011_00 — Out-of-Distribution (not seen during training)

| Model | Thr | Mean Err | P@3px | P@5px | Avg Matches |
|-------|-----|----------|-------|-------|-------------|
| Baseline (pretrained) | 0.005   | 97.36px | 0.01% | 0.03% | 2256 |
| Run 13 (fine-tuned)   | 0.0001  | 73.75px | 0.40% | 1.04% | 1136 |
| Run 13 (fine-tuned)   | 0.001   | 72.67px | 0.44% | 1.12% | 889  |
| Run 13 (fine-tuned)   | 0.002   | 73.47px | 0.45% | 1.16% | 525  |
| **Run 13 (fine-tuned)**   | **0.005**   | **75.96px** | **0.47%** | **1.25%** | **159** |
| Run 13 (fine-tuned)   | 0.01    | 77.93px | 0.14% | 0.45% | 50   |
| Run 13 (fine-tuned)   | 0.02    | 72.38px | 0.00% | 0.00% | 18   |

**Finding**: Fine-tuning improved mean error by ~1.3× on OOD (97px → 76px) and P@3px by ~47× (0.01% → 0.47%).
**Optimal threshold is 0.005** for OOD as well — same pattern as in-distribution.
However, OOD performance (76px) is much worse than in-distribution (20px), confirming a significant generalization gap.

---

## Key Observations & Lessons

1. **`num_matches` is a useless training metric** — always fixed at 7680 in training mode due to GT padding in `CoarseMatching.get_coarse_match()`. Only `val/num_matches` reflects real predicted matches.

2. **`conf_max` is a collapse detector, not a quality metric** — dual-softmax always produces a peak somewhere regardless of match quality. Only useful for detecting uniform collapse.

3. **Triplet loss with pretrained model → zero gradients** — the pretrained model already satisfies `sim_pos - sim_neg > 1.0` for most pairs. Focal loss is necessary to always produce non-zero gradients.

4. **Baseline has terrible raw accuracy** — 92px mean error and 0.03% P@3px suggests the vanilla pretrained model does not generalize well on this evaluation protocol. Fine-tuning improved accuracy significantly but absolute numbers remain low.

5. **Generalization gap** — scene0011 (unseen) performs much worse than scene0000 (seen). Training on a single scene overfits to that scene's appearance distribution.

---

## Training Configuration (Run 13)

| Parameter | Value |
|-----------|-------|
| Steps | 20,000 |
| Batch size | 8 |
| Learning rate | 1e-5 |
| Optimizer | AdamW (weight_decay=1e-4) |
| Scheduler | CosineAnnealingLR (eta_min=1e-6) |
| Gradient clip | 1.0 |
| Precision | bf16-mixed |
| Tau (epipolar) | 50.0 |
| Frame gap | 20 |
| Scenes | 1 (scene0000_00) |
| Checkpoint dir | cr13 |
