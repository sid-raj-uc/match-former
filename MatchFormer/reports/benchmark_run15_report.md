# Benchmark Report — epipolar-step=50000.ckpt (Run 15)

Comparison of run13 (20k steps) vs run15 (50k steps from scratch) checkpoints.
Evaluated on 100 pairs, frame_gap=20, tau=50.0, threshold sweep [0.0001 → 0.2].

**Checkpoint**: `epipolar-step=50000.ckpt`
**Training config**: 50k steps, batch=8, lr=1e-5, FocalLoss (α=0.5, γ=2.0),
selective freeze (AttentionBlock3+4 + fine FPN head, 64.9% trainable), 10 scenes (0000–0010), from scratch.

---

## scene0000 — In-Distribution (seen during training)

### Run 15 — epipolar-step=50000.ckpt

| Thr    | Mean Err (px) | P@3px  | P@5px  | Avg Matches |
|--------|--------------|--------|--------|-------------|
| 0.0001 | 10.22        | 8.30%  | 20.75% | 2387        |
| 0.0005 | 10.22        | 8.30%  | 20.75% | 2387        |
| 0.001  | 10.21        | 8.30%  | 20.75% | 2387        |
| 0.002  | 10.19        | 8.31%  | 20.78% | 2384        |
| 0.005  | 10.10        | 8.39%  | 20.99% | 2341        |
| 0.01   | 9.79         | 8.81%  | 21.94% | 2085        |
| 0.02   | 9.25         | 9.73%  | 23.83% | 1376        |
| **0.05**   | **8.64**     | **11.09%** | **27.03%** | **228** |
| 0.10   | 9.39         | 11.78% | 28.36% | 12          |
| 0.20   | 9.80         | 0.00%  | 0.00%  | 1           |

**Recommended threshold: 0.05** — best mean error (8.64px) with healthy match count (228).
At thr=0.10, P@3px peaks at 11.78% but only 12 matches — too few to be reliable.

### Head-to-Head: Run13 vs Run15 at key thresholds

| Thr   | Run13 Mean Err | Run13 P@3px | Run15 Mean Err | Run15 P@3px | Run15 Matches |
|-------|---------------|-------------|---------------|-------------|--------------|
| 0.005 | 20.47px       | 1.84%       | **10.10px**   | **8.39%**   | 2341         |
| 0.02  | 26.47px       | 0.52%       | **9.25px**    | **9.73%**   | 1376         |
| 0.05  | 29.03px       | 0.31%       | **8.64px**    | **11.09%**  | 228          |

**Run15 improvement on scene0000:**
- Mean error: **2× better** (20.47px → 10.10px at thr=0.005)
- P@3px: **4.5× better** (1.84% → 8.39% at thr=0.005)
- Optimal threshold shifted from 0.005 → 0.05 (model produces more high-confidence correct matches)

---

## scene0011 — Out-of-Distribution (not seen during training)

### Run 15 — epipolar-step=50000.ckpt

| Thr    | Mean Err (px) | P@3px | P@5px | Avg Matches |
|--------|--------------|-------|-------|-------------|
| 0.0001 | 65.32        | 0.15% | 0.46% | 1717        |
| 0.0005 | 64.94        | 0.15% | 0.46% | 1715        |
| 0.001  | 64.65        | 0.15% | 0.46% | 1710        |
| 0.002  | 64.39        | 0.15% | 0.47% | 1683        |
| 0.005  | 64.29        | 0.18% | 0.54% | 1464        |
| 0.01   | 64.86        | 0.20% | 0.67% | 1012        |
| **0.02**   | **66.53**    | **0.22%** | **0.84%** | **453** |
| 0.05   | 61.39        | 0.00% | 0.00% | 72          |
| 0.10   | 64.75        | 0.00% | 0.00% | 35          |
| 0.20   | 60.80        | 0.00% | 0.00% | 15          |

**Recommended threshold: 0.02** — best P@3px (0.22%) with 453 matches.

### Head-to-Head: Run13 vs Run15

| Thr   | Run13 Mean Err | Run13 P@3px | Run15 Mean Err | Run15 P@3px | Run15 Matches |
|-------|---------------|-------------|---------------|-------------|--------------|
| 0.005 | 75.96px       | 0.47%       | 64.29px       | 0.18%       | 1464         |
| 0.01  | 77.93px       | 0.14%       | 64.86px       | 0.20%       | 1012         |
| 0.02  | 72.38px       | 0.00%       | 66.53px       | 0.22%       | 453          |

**Run15 on scene0011:**
- Mean error improved: **76px → 64px (~1.2×)**
- P@3px: mixed — run13 had 0.47% at thr=0.005, run15 peaks at 0.22% at thr=0.02
- Run15 outputs far more matches (1464 vs 159 at thr=0.005) — model is less conservative on OOD

---

## Full Comparison: Baseline vs Run13 vs Run15

### scene0000 (in-distribution, thr=0.005)

| Model | Mean Err | P@3px | P@5px | Matches |
|-------|----------|-------|-------|---------|
| Baseline (pretrained) | 92.53px | 0.03% | 0.08% | 2723 |
| Run 13 (20k steps)    | 20.47px | 1.84% | 4.88% | 439  |
| **Run 15 (50k steps)**    | **10.10px** | **8.39%** | **20.99%** | **2341** |

### scene0011 (OOD, thr=0.005)

| Model | Mean Err | P@3px | P@5px | Matches |
|-------|----------|-------|-------|---------|
| Baseline (pretrained) | 97.36px | 0.01% | 0.03% | 2256 |
| Run 13 (20k steps)    | 75.96px | 0.47% | 1.25% | 159  |
| Run 15 (50k steps)    | 64.29px | 0.18% | 0.54% | 1464 |

---

## Key Observations

1. **50k steps dramatically improves in-distribution performance** — P@3px jumps from 1.84% to 8.39% on scene0000, mean error halves (20px → 10px). Training longer on 10 diverse scenes pays off.

2. **Optimal threshold shifted** — run13 peaked at thr=0.005, run15 peaks at thr=0.05. The 50k model learned to assign high confidence more selectively, so a higher threshold filters to better matches.

3. **OOD mean error improved but P@3px dropped** — scene0011 mean error went 76px → 64px, but P@3px at thr=0.005 dropped from 0.47% to 0.18%. Run15 outputs many more OOD matches (1464 vs 159) but they're less precise — the model is more "open" to matching on unseen scenes.

4. **Generalization gap persists** — scene0000 P@3px at 8.39% vs scene0011 at 0.22% (38× gap). More training reduced the in-distribution error but didn't close the OOD gap proportionally.

5. **Run15 match count behavior** — at thr=0.005, run15 outputs 2341 matches on scene0000 vs 439 for run13. The model learned to be confidently correct on in-distribution data, producing many high-quality matches.
