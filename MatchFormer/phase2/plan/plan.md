# Epipolar Curriculum: Revised Training Plan

## Core Thesis

Use the epipolar mask as a geometric scaffold during early training, then systematically anneal it away while introducing an epipolar loss to verify the network has internalized the constraint. The mask teaches fast, the loss verifies deep. At inference, no mask, no intrinsics, no pose prior — just a network that has learned geometric reasoning from the curriculum.

---

## Architecture Baseline

This plan assumes the current Matchformer architecture and training setup:

| Component              | Detail                                                        |
|------------------------|---------------------------------------------------------------|
| Backbone               | Frozen except AttentionBlock3, AttentionBlock4, FPN head      |
| BatchNorm              | **Always frozen in eval mode** — never updated, all phases    |
| Coarse matching        | Dual-softmax on similarity matrix S                           |
| Epipolar mask          | Soft Gaussian, applied **after** dual-softmax on confidence P |
| Fine matching          | Subpixel regression from coarse match proposals               |
| Loss                   | Weighted combination of L_coarse + L_fine                     |
| Training data          | 11-scene dataset (~24k frames), batch size 4                  |

### Critical architectural note: mask placement

The epipolar mask is applied **after** dual-softmax, not before. This means the softmax normalization operates on the full, unmasked similarity matrix S. The mask then downweights off-epipolar entries in the resulting confidence matrix P:

```
S(i, j) = <f₁(i), f₂(j)>                    # raw similarity
P(i, j) = softmax_row(S) · softmax_col(S)     # dual-softmax (full distribution)
P_masked(i, j) = P(i, j) · M(i, j)           # mask applied post-softmax
```

Why this matters for the curriculum: because the attention mechanism sees the full similarity landscape during the forward pass, it is already learning to process the complete distribution — the mask only corrects the output. When the mask is removed in Phase 3, the distribution shift is much less violent than it would be if the mask were applied before softmax. This is the safer design for progressive annealing.

---

## Training Infrastructure

### Optimizer: per-phase cosine schedules

The original plan used a single CosineAnnealingLR over 50k steps. This creates a commitment problem — if Phase 1 ends early or fails, the LR budget for later phases is distorted.

**Revised approach:** Use CosineAnnealingWarmRestarts or separate CosineAnnealingLR instances per phase. Each phase gets its own full cosine cycle. If a phase ends early, no LR budget is wasted.

| Phase   | Steps       | Base LR | eta_min | T_max (cosine period) |
|---------|-------------|---------|---------|-----------------------|
| Phase 1 | 0 – ~10k    | 3e-5    | 5e-6    | = Phase 1 duration    |
| Phase 2 | ~10k – ~25k | 2e-5    | 3e-6    | = Phase 2 duration    |
| Phase 3 | ~25k – ~40k | 1e-5    | 1e-6    | = Phase 3 duration    |

Key changes from original:

- **Base LR raised to 3e-5 for Phase 1.** With only Block3/4 + FPN head unfrozen, 1e-5 is too conservative. The small trainable parameter set needs a higher LR to move meaningfully. Each subsequent phase drops the base LR as the network fine-tunes more delicately.
- **eta_min raised to 5e-6 for Phase 1.** At 1e-6 with few trainable params, the network is barely learning by the end of the cosine cycle.
- **Step counts are approximate.** Phase transitions are gated by diagnostics (GIR), not fixed step counts. The step ranges here are estimates based on your dataset size and observed convergence behavior.

### BatchNorm: frozen, always

```python
# Must be called before every training loop iteration, or once if using a wrapper
model.backbone.eval()  # Keeps BN in eval mode even when model is in train mode
```

This is non-negotiable across all three phases. With batch size 4, BN statistics would drift catastrophically if updated. The backbone BN layers use their pretrained running mean/variance throughout.

### Data pipeline: frame gap and hard negatives

The original plan omitted this entirely. This is critical.

**Frame gap curriculum:**

The frame gap controls pair difficulty. Too small = trivial pairs, loss collapse. Too large = insufficient overlap, matches are sparse and noisy.

| Phase   | Frame gap strategy                                                  |
|---------|---------------------------------------------------------------------|
| Phase 1 | Start at gap=20 (known stable from your experiments). Hold constant. |
| Phase 2 | Gradually increase to gap=30–40 as mask anneals. Harder pairs force the network to rely on internalized geometry rather than easy photometric matching. |
| Phase 3 | Use mixed gaps (20–50, sampled uniformly). The network must handle varying difficulty without the mask. |

**Hard negative sampling within the epipolar band:**

With the mask active, the trivial distractors (off-epipolar matches) are already suppressed. The hard negatives that actually matter are **on-epipolar false matches**: points that lie on the correct epipolar line but correspond to the wrong 3D point. These arise from repeated textures, similar-looking patches at different depths, and symmetric structures.

Strategy: for each ground-truth match (pᵢ, pⱼ), identify candidate pixels along the epipolar line lᵢ in I₂ that have high feature similarity to pᵢ but are NOT the correct match. These are the on-epipolar hard negatives. Ensure the training batch includes a sufficient proportion of image pairs where such ambiguities exist (e.g., scenes with repetitive textures, corridors, tiled floors).

This is most important in Phase 1 where the mask makes the problem too easy on simple pairs, and in Phase 2 where the network needs to learn to disambiguate without the mask's help.

---

## Phase 1: Scaffolded Training

**Duration:** ~8k–12k steps (adaptive, gated by convergence + GIR diagnostic)  
**Mask:** Full strength (σ = σ₀, constant)  
**Epipolar loss:** Inactive (weight = 0)  
**LR schedule:** CosineAnnealingLR, base=3e-5, eta_min=5e-6, T_max=Phase 1 duration  
**Frame gap:** 20 (constant)

### What happens

The soft Gaussian epipolar mask is applied at full strength after dual-softmax. For each query pixel pᵢ in I₁, the mask computes the epipolar line lᵢ = F · p̃ᵢ in I₂, then applies a Gaussian weighting based on perpendicular distance:

```
M(i, j) = exp( -d(pⱼ, lᵢ)² / (2σ₀²) )
```

The mask is applied to the confidence matrix: P_masked = P · M. The loss is the standard coarse + fine combination.

### What the network learns

The mask constrains the effective search space, so the network focuses on learning discriminative appearance features within the epipolar band. Cross-attention patterns are shaped by the mask into stripe-like, epipolar-consistent patterns. The open question is how much of this geometric structure gets internalized into the weights versus externally imposed by the mask.

### What the network does NOT learn

Because the mask handles geometric filtering, the network has no incentive to reject geometrically inconsistent matches on its own, handle degenerate configurations (pure rotation, planar scenes), or match without calibration. These are Phase 2 and 3 objectives.

### When to stop Phase 1

Do NOT default to a fixed step count. Monitor two signals:

1. **Validation loss plateau:** When mask-on validation loss stops improving for 2k+ steps, the network has extracted what it can from masked training.
2. **Training loss collapse warning:** If training loss collapses rapidly (within first 2k–3k steps), the mask + frame_gap combination is making the problem too easy. Either increase frame_gap or proceed to Phase 2 sooner.

With your 11-scene dataset at batch 4, expect convergence somewhere in the 8k–12k range. The 20k number from the original plan was too conservative.

### Phase 1 exit diagnostic

Run the checkpoint in two modes on a held-out validation set:

**Mode A — Mask ON** (standard inference)  
**Mode B — Mask OFF** (M(i,j) = 1 for all i,j)

Compute for both modes:

1. **Epipolar residual distribution:** For each predicted match, compute the Sampson distance. Plot histograms for both modes.

2. **Pose estimation AUC:** Run RANSAC on predicted matches, estimate relative pose, report AUC@5°, @10°, @20°.

3. **Geometric Internalization Ratio (GIR):**

```
GIR = AUC_mask_off(@5°) / AUC_mask_on(@5°)
```

4. **Attention map visualization:** For 5–10 representative pairs, visualize cross-attention weights for a selected query pixel. Save mask-ON and mask-OFF side by side. Look for stripe-like patterns in the mask-OFF case.

### GIR interpretation and Phase 2 calibration

| GIR       | Meaning                          | Phase 2 annealing rate α |
|-----------|----------------------------------|--------------------------|
| > 0.8     | Strong internalization           | α = 3–4 (aggressive)     |
| 0.5 – 0.8 | Partial internalization          | α = 2–3 (moderate)       |
| < 0.5     | Weak internalization             | α = 1–2 (slow)           |
| < 0.3     | Network is mask-dependent        | Extend Phase 1 or increase σ₀ |

### Failure modes

**Loss collapses immediately:** Mask + small frame gap + easy scenes = trivial problem. Increase frame_gap to 30+, or add harder scene pairs to the training set.

**Mask-OFF performance is near zero (GIR ≈ 0):** σ₀ was too small. The network only ever saw a razor-thin band and learned nothing about broader geometric structure. Increase σ₀ (try 2× current value) and retrain.

**Loss is noisy, doesn't converge smoothly:** Calibration noise in some image pairs is corrupting the mask. Filter pairs by median SIFT epipolar distance; exclude pairs where this exceeds 5 pixels at full resolution.

**Loss spikes on specific pairs:** Degenerate geometry (pure rotation, planar scenes). Detect these during data loading by checking the condition number of F or baseline-to-depth ratio. Exclude them or disable the mask for those pairs only.

---

## Phase 2: Annealing + Loss Handoff

**Duration:** ~12k–15k steps (adaptive)  
**Mask:** Annealing from σ₀ toward ∞ (exponential schedule, rate α set by GIR)  
**Epipolar loss:** Ramps up from 0 to λ_max  
**LR schedule:** CosineAnnealingLR, base=2e-5, eta_min=3e-6, T_max=Phase 2 duration  
**Frame gap:** Gradually increasing from 20 to 30–40

### The two simultaneous processes

**Process 1: Mask annealing**

The mask bandwidth σ grows exponentially, making the Gaussian progressively wider and the constraint progressively weaker:

```
σ(t) = σ₀ · exp(α · (t - t_phase2_start) / T_phase2)
```

where α is the annealing rate (set by GIR from Phase 1 diagnostic) and T_phase2 is the Phase 2 duration. As σ → ∞, the Gaussian becomes flat, M(i,j) → 1 for all (i,j), and the mask has no effect.

The exponential schedule gives a long period of "mostly constrained" followed by rapid release. This is gentler than linear annealing, which removes the constraint too fast early on.

**Process 2: Epipolar loss ramp-up**

Simultaneously, a SCENES-style epipolar regression loss is introduced on the fine matching stage. For each predicted fine match (pᵢ, pⱼ), compute the Sampson distance:

```
d_sampson = (p̃ⱼᵀ F p̃ᵢ)² · [ 1/||(Fp̃ᵢ)₁:₂||² + 1/||(Fᵀp̃ⱼ)₁:₂||² ]
```

The epipolar loss is the mean Sampson distance over all predicted fine matches:

```
L_epi = (1/N) · Σ d_sampson(pᵢ, pⱼ)
```

Its weight ramps linearly over the first half of Phase 2:

```
λ_epi(t) = λ_max · min(1, (t - t_phase2_start) / (T_phase2 / 2))
```

The total loss in Phase 2 becomes:

```
L_total = w_coarse · L_coarse + w_fine · L_fine + λ_epi(t) · L_epi
```

### Why both at the same time

The mask constrains what the network **sees** (input filtering). The loss constrains what the network **produces** (output penalization). As the mask relaxes, the network loses its geometric input filter. The epipolar loss catches it if outputs start drifting geometrically. It's like removing scaffolding from a building while running structural load tests at each floor.

### Frame gap escalation

As the mask anneals, gradually increase frame_gap from 20 toward 30–40. The rationale: with a weaker mask, the network needs harder pairs to force it to develop its own geometric discrimination. Easy pairs at gap=20 with a weak mask give the network no reason to internalize geometry — it can still match by appearance alone.

```
frame_gap(t) = 20 + 20 · (t - t_phase2_start) / T_phase2
```

This linearly increases the gap from 20 to 40 over Phase 2. Adjust the range based on your dataset's overlap characteristics.

### Monitoring the handoff

Every 2k steps during Phase 2, compute the GIR (same protocol as Phase 1 exit diagnostic):

```
GIR(t) = AUC_mask_off(@5°) / AUC_mask_on(@5°)
```

Plot GIR over Phase 2. It should be **monotonically increasing** toward 1.0. If it plateaus or drops:

- **GIR plateaus:** The network has hit a ceiling on geometric internalization at the current annealing rate. The mask is being removed faster than the network can compensate. Slow down: reduce α by 0.5 and continue.
- **GIR drops:** The network is regressing — it's losing geometric ability as the mask weakens. This is the critical failure mode. Pause annealing (hold σ constant), increase λ_epi, and let the epipolar loss stabilize the network before resuming.
- **Training loss spikes:** Same cause as GIR drop. The distribution shift from mask removal is too abrupt. Slow the annealing.

### Phase 2 exit criteria

Phase 2 is complete when:

1. The mask has fully annealed (σ is large enough that M ≈ 1 everywhere, effectively no mask)
2. GIR > 0.85 (the network performs nearly as well without the mask as with it)
3. Training loss has restabilized after the mask removal

If GIR doesn't reach 0.85 by the end of Phase 2's step budget, extend Phase 2 or revisit Phase 1's σ₀ and training setup.

---

## Phase 3: Free Flight

**Duration:** ~10k–15k steps  
**Mask:** Completely removed (M = 1)  
**Epipolar loss:** Decaying from λ_max to a light residual  
**LR schedule:** CosineAnnealingLR, base=1e-5, eta_min=1e-6, T_max=Phase 3 duration  
**Frame gap:** Mixed (20–50, uniformly sampled)

### What happens

The mask is gone. The network's coarse matching runs exactly as it will at inference — pure dual-softmax on unmasked similarity scores. The only remaining geometric signal is the epipolar loss, which decays:

```
λ_epi(t) = λ_max · max(0.1, 1 - (t - t_phase3_start) / (T_phase3 · 0.75))
```

This decays the loss weight from λ_max to 0.1 · λ_max over the first 75% of Phase 3, then holds at 0.1 · λ_max for the remaining 25%. The residual prevents geometric drift during the final training steps.

### Why keep a residual epipolar loss

Without any geometric signal, extended fine-tuning can cause the network to "forget" the geometric patterns it internalized — especially when trained on diverse, mixed-gap pairs. The light residual (10% of peak weight) acts as a gentle regularizer. It doesn't constrain the network enough to hurt on edge cases (moving objects, degenerate geometry), but it maintains a baseline geometric bias.

### Mixed frame gap

Phase 3 uses uniformly sampled frame gaps from 20 to 50. The network must handle varying difficulty levels without any mask assistance. This is the regime closest to real-world deployment where you don't control the baseline between views.

### What the network should look like at the end

Visualize cross-attention maps at the final checkpoint. Without any mask, the attention patterns should show stripe-like structures roughly aligned with where the epipolar lines would be — emergent geometric reasoning baked into the weights. This is the visual proof that the curriculum worked.

### Final evaluation protocol

Run the final checkpoint on three evaluation settings:

**Setting 1 — Calibrated (same domain).** Validation split of training data. Compare against the Phase 1 mask-ON model. If Phase 3 matches or exceeds Phase 1, the curriculum successfully transferred geometric reasoning from the mask into the weights.

**Setting 2 — Uncalibrated (same domain).** Same validation data, but no intrinsics or pose provided at inference. Phase 1 model cannot run here (it needs F for the mask). Phase 3 model can, because it has no mask. This demonstrates the generalizability payoff.

**Setting 3 — Out-of-domain.** A dataset the model has never seen during training — different camera, different environment, different characteristics. This tests whether the internalized geometry is general or overfit to the training distribution.

Report AUC@5°/10°/20° for all three settings. The headline result: a model trained with geometric scaffolding that deploys without it, matching or exceeding the scaffolded model's performance.

---

## Ablation Study Design

To make this publishable, run four training configurations on the same data with the same total step budget:

| Condition         | Description                                          |
|-------------------|------------------------------------------------------|
| **Mask-only**     | Your current approach. Mask at full strength, never removed. Baseline. |
| **Loss-only**     | No mask at any point. SCENES-style epipolar loss from step 0. This is the SCENES approach. |
| **Mask + Loss**   | Mask at full strength + epipolar loss, both active the entire time. No annealing. |
| **Curriculum**    | The full phased plan: mask → anneal → loss handoff → free flight. |

The curriculum condition should outperform all three others on the uncalibrated and out-of-domain settings, because it's the only one that both leverages the mask's strong geometric guidance AND learns to function without it.

Mask-only will likely win on calibrated in-domain (it has the mask at inference). Loss-only will likely underperform on all settings because the loss signal is weaker than the mask during early training. Mask + Loss (no annealing) will be strong in-domain but can't deploy uncalibrated.

---

## Implementation Checklist

Before starting training:

- [ ] Confirm mask is applied **after** dual-softmax, not before
- [ ] Confirm BN is frozen in eval mode (add explicit check in training loop)
- [ ] Set up per-phase LR scheduling (separate cosine cycles, not one global schedule)
- [ ] Implement frame_gap as a configurable parameter that can change between phases
- [ ] Implement σ annealing function for Phase 2 (exponential growth of mask bandwidth)
- [ ] Implement Sampson distance computation for the epipolar loss
- [ ] Set up GIR evaluation script (mask-on vs mask-off AUC comparison)
- [ ] Set up attention map visualization for qualitative monitoring
- [ ] Filter training pairs by calibration quality (exclude pairs with high median SIFT epipolar distance)
- [ ] Identify and flag degenerate geometry pairs (pure rotation, planar scenes)

Before each phase transition:

- [ ] Run GIR diagnostic on current checkpoint
- [ ] Save attention map visualizations
- [ ] Log all metrics for comparison across phases
- [ ] Verify training loss is stable (no spikes or collapse)

---

## Expected Timeline

With your hardware and dataset:

| Phase   | Steps     | Wall time estimate | Key output               |
|---------|-----------|--------------------|--------------------------|
| Phase 1 | 8k–12k   | ~3–5 hours         | GIR baseline + checkpoint |
| Phase 2 | 12k–15k  | ~5–7 hours         | GIR > 0.85 + checkpoint   |
| Phase 3 | 10k–15k  | ~4–6 hours         | Final model + evaluation  |
| Ablations| 4× above | ~2–3 days total    | Comparison table          |

Total: roughly 30k–42k steps for the curriculum run, plus 3× that for the ablation conditions. Budget approximately one week of GPU time for the full experiment including ablations.