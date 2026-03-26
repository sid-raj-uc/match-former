# MatchFormer Epipolar Fine-Tuning — Project History

## Goal

Fine-tune MatchFormer (indoor-lite-LA pretrained checkpoint) on ScanNet indoor scenes to improve coarse feature matching under epipolar constraints. The model uses dual-softmax over a similarity matrix to produce a confidence matrix, then performs mutual nearest-neighbor filtering to select matches.

**Metric**: pixel reprojection error on predicted fine matches (depth-projected GT). Lower = better.
**Key metrics**: Mean Err (px), P@3px (% of matches within 3px of GT), P@5px, Avg Matches.

---

## Baseline (pretrained `indoor-lite-LA.ckpt`, no fine-tuning)

Evaluated on OOD scenes (scenes 0011, 0012, 0013 — not used in training):

| Scene | Mean Err | P@3px | P@5px | Avg Matches |
|-------|----------|-------|-------|-------------|
| scene0011 | 97.36px | 0.01% | 0.03% | 2256 |
| scene0012 | 222.13px | 0.00% | 0.00% | 2612 |
| scene0013 | 199.00px | 0.00% | 0.00% | 2426 |

Adding epipolar constraint at inference time on the pretrained model (vanilla+epipolar, tau=50) gives negligible improvement — confidence scores carry no geometric information, so the mask just removes a few random matches.

---

## Architecture

- **Model**: MatchFormer indoor-lite-LA (20.26M params total)
- **Trainable subset** (selective freeze): AttentionBlock3, AttentionBlock4, fine FPN head (`layer1_outconv`, `layer1_outconv2`) — 13.14M / 20.26M params (64.9%)
- **Coarse matching**: dual-softmax `conf_matrix = softmax(sim, dim=1) * softmax(sim, dim=2)`, then mutual nearest-neighbor + threshold
- **Epipolar injection** at training and inference: multiply `conf_matrix` by a soft Gaussian epipolar mask (`exp(-dist/tau)`) computed from the fundamental matrix F derived from GT poses
- **Supervision**: depth-projected GT correspondences (`compute_supervision`) — reprojects depth-unprojected 3D points through relative pose to find GT coarse patch pairs (`spv_b_ids`, `spv_i_ids`, `spv_j_ids`)
- **Fine loss**: L2 on predicted fine-level offset vs GT zero-offset for GT-padded matches, weighted by inverse predicted std

---

## Training Config (all runs)

- **Data**: ScanNet scenes 0000–0010 (11 scenes), frame_gap=20
- **Batch size**: varied (2–8)
- **Optimizer**: AdamW + CosineAnnealingLR
- **Gradient clip**: 1.0
- **Colab** (Google Drive checkpoints, T4/L4 GPU)

---

## Experiments Chronology

### Phase 0 — Infrastructure Setup
Built the full training pipeline from scratch:
- `train_finetune.py`: monkey-patches `CoarseMatching.forward` to inject epipolar mask per batch
- `EpipolarFineTuner`: computes F matrix from batch poses (OpenGL→OpenCV conversion) before each step
- `ScanNetSimpleDataset`: loads color images, depth, intrinsics, poses from exported ScanNet scans
- `run_benchmark.py`: threshold sweep over 100 pairs, vanilla vs vanilla+epipolar comparison
- `supervision.py`: GT correspondence generation via depth unprojection + reprojection

### Phase 1 — Basic Focal Loss Training

**Loss**: Binary focal loss on `conf_matrix` (dual-softmax output), all negatives
`FL(p) = -alpha * (1-p)^gamma * log(p)`

Several hyperparameter sweeps via git commits:
- `31aa261` — tried different alpha/gamma combinations
- `0bf77ed` — trying new config
- `b884adf`, `ce857ca` — added new loss variants

**Problem encountered**: Confidence collapse. Dual-softmax over ~4800 entries (e.g. 60×80 grid) means the maximum possible confidence value is `softmax_max^2 ≈ (1/1)*(1/1)` in theory but in practice values stay very small. The focal loss on these values is permanently unsatisfied and generates destabilizing gradients that flatten the confidence distribution → all matches have near-equal confidence → threshold filtering becomes meaningless.

### Phase 2 — Sampled Negatives to Control Gradient Ratio

**Loss**: Focal loss with `neg_per_pos` sampling — instead of all ~23M negatives, sample N negatives per positive from the same row and column.

Commits: `1d80453`, `c42947e` "adding random 15 vectorized", `b8ba437` "changing lr and changing gradient decrease only to 15"

**Rationale**: With all negatives, the positive-to-negative gradient ratio is ~1:4800, overwhelming positive signal. Sampling 15 per positive gives a 1:30 ratio.

**Result**: Helped stabilize training to some extent but did not fundamentally fix confidence collapse because the underlying issue (loss on dual-softmax values that can't reach 1.0) remained.

### Phase 3 — KL Divergence Loss

Commits: `1979fc4`, `5cea49e` "adding kl divergence"

**Idea**: Instead of focal loss treating each cell independently, use KL divergence to push the confidence distribution toward a target peaky distribution around GT matches.

**Result**: Did not improve meaningfully. Abandoned.

### Phase 4 — Weight Freezing Experiments

Commits: `1a94a4e` "added model weight freeze", `66cd1fe` "trained with frozen weights all except the matching layers"

**Idea**: Freeze more of the network to prevent catastrophic forgetting of pretrained features. Only train the final attention blocks and matching head.

**Variants tried**:
- Freeze all except coarse matching layer
- Freeze all except AttentionBlock3+4 + fine FPN head (current default, 64.9% trainable)

### Phase 5 — Run15: Working Configuration

**Checkpoint**: `epipolar-step=50000.ckpt`
**Config**: 50k steps, batch=8, lr=1e-5, FocalLoss (α=0.5, γ=2.0), selective freeze (AttentionBlock3+4 + fine FPN head), 10 scenes (0000–0010), tau=50.0

This is the best checkpoint produced so far.

**OOD Results at thr=0.005**:

| Scene | Mean Err | P@3px | P@5px | Avg Matches | vs Baseline |
|-------|----------|-------|-------|-------------|-------------|
| scene0011 | 64.29px | 0.18% | 0.54% | 1464 | 3–5× better |
| scene0012 | 25.77px | 0.47% | 1.44% | 1969 | ~8× better |
| scene0013 | 26.67px | 1.34% | 3.28% | 1638 | ~7× better |

**In-distribution (scene0000) at thr=0.05**: 8.64px mean err, 11.09% P@3px — confirms the model learned something.

**Remaining gap**: scene0000 (in-dist) is ~8× better than OOD average. scene0011 is notably harder than 0012/0013 (64px vs 26px mean err), suggesting appearance distribution shift.

Commits: `d6903e0` "unfrozen weights added", `4d94e0a` "fix"

### Phase 6 — Triplet Loss Attempt (current, not yet evaluated)

**Problem diagnosis**: The core issue is **confidence collapse** — the focal loss on dual-softmax values is permanently unsatisfied because dual-softmax over large grids (e.g. 60×80=4800 entries) can never approach 1.0, so gradients never go to zero and keep flattening the distribution.

**Fix attempted**: Replace focal loss on `conf_matrix` with triplet margin loss on the raw pre-softmax `sim_matrix`.

```
loss = mean( max(0, margin - sim[i, j_pos] + sim[i, j_neg]) )
```

Once every triplet satisfies the margin, gradient = 0 and features stabilize.

**Implementation**:
- `losses.py`: `FocalLoss` → `TripletCoarseLoss(margin=1.0, neg_per_pos=10)`. Picks top-20 hard negatives per positive, randomly samples 10, computes hinge loss.
- `train_finetune.py`: `data.update({'conf_matrix': conf_matrix, 'sim_matrix': sim_matrix})` — stores raw sim before softmax
- `lightning_loftr.py`: clears `sim_matrix` key before training forward pass

**Status**: Implemented but not yet benchmarked. User reports "didn't improve anything" — need training logs to diagnose (is `loss_c` going to zero? is `conf_max` still collapsing?).

---

## Key Observations

1. **Fine-tuning is essential** — epipolar mask at inference on pretrained model gives <1% improvement. The pretrained confidence scores are uniformly bad and carry no geometric signal.

2. **Selective freezing was important** — training all weights led to catastrophic forgetting. Freezing the backbone and only training AttentionBlock3+4 + fine FPN head gave the best results.

3. **LR matters** — 1e-5 worked better than higher LRs for this fine-tuning regime.

4. **Confidence collapse is the central unsolved problem** — the dual-softmax architecture is fundamentally at odds with cross-entropy-style losses when the grid is large. The positive confidence values can never approach 1.0, so the loss is permanently unsatisfied.

5. **Optimal threshold varies per scene** — scene0011 peaks at thr=0.02, scenes 0012/0013 peak at thr=0.05. The model's confidence calibration is not uniform across OOD scenes.

6. **Vanilla baseline is threshold-insensitive** — pretrained model's confidence scores carry no information about match quality (metrics don't change across 0.0001–0.20 range), confirming total confidence collapse in the pretrained model on this domain.

7. **Generalization gap persists** — roughly 8× worse on OOD vs in-distribution. More diverse training data or domain randomization would be needed to close this.

---

## Current Code State

### Key Files

| File | Purpose |
|------|---------|
| `MatchFormer/train_finetune.py` | Training launcher, `EpipolarFineTuner`, epipolar forward patch |
| `MatchFormer/model/lightning_loftr.py` | `PL_LoFTR` base class, training/validation step, freeze logic |
| `MatchFormer/model/losses.py` | `TripletCoarseLoss` (current), `FocalLoss` (deprecated but still in file), `MatchFormerLoss`, `fine_loss` |
| `MatchFormer/model/supervision.py` | GT correspondence generation from depth+poses |
| `MatchFormer/model/backbone/coarse_matching.py` | Dual-softmax matching, MNN filtering, GT padding for training |
| `MatchFormer/run_benchmark.py` | Threshold sweep benchmark |
| `MatchFormer/model/datasets/scannet_simple.py` | ScanNet data loader |

### Current Loss Architecture

```python
# MatchFormerLoss.forward:
loss_c = TripletCoarseLoss(margin=1.0, neg_per_pos=10)(sim_matrix, spv_b, spv_i, spv_j)
loss_f = fine_loss(data)  # L2 on fine-level offset
total = 1.0 * loss_c + 0.5 * loss_f
```

### Training Command

```bash
python train_finetune.py \
  --data_dir ../data/scans \
  --ckpt model/weights/indoor-lite-LA.ckpt \
  --steps 50000 \
  --lr 1e-5 \
  --batch 8 \
  --tau 50.0 \
  --frame_gap 20 \
  --wandb
```

---

## Things NOT Yet Tried

- Training on more diverse scenes (more than 11)
- Contrastive loss (InfoNCE/NTXent) on patch descriptors directly
- Curriculum: start with easy pairs (small frame gap) and increase
- Augmentation: photometric jitter, random crop, scale jitter
- Cosine similarity normalization before triplet (L2-normalize features before dot product)
- Different margins for triplet (currently 1.0 — may be too large or too small)
- Logging `sim_matrix` stats (mean, std, pos/neg separation) to understand what the triplet loss is doing to the feature space
- Running the triplet loss with a debug check: print `(sim_pos - sim_neg).mean()` to see if triplets are already satisfied at initialization (which would mean zero gradient from step 1)
