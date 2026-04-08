# Project Proposal: Epipolar Curriculum Fine-Tuning for Robust Feature Matching on Out-of-Distribution Datasets

## 1. Background and Prior Work

Feature matching is a foundational task in computer vision, enabling downstream applications such as visual odometry, SLAM, structure-from-motion, and 3D reconstruction. Transformer-based matchers like MatchFormer and LoFTR have achieved strong performance on standard indoor (ScanNet) and outdoor (MegaDepth) benchmarks by interleaving self- and cross-attention within a coarse-to-fine matching pipeline.

However, these models are trained on curated datasets with well-calibrated cameras, moderate baselines, and static scenes. When deployed on o
ut-of-distribution (OOD) data — such as the aggressive drone trajectories in EuRoC-MAV or the extreme viewpoint and illumination changes in WxBS — performance degrades significantly. The pretrained models produce fewer matches, higher reprojection errors, and worse pose estimation accuracy.

### What We Have Done So Far

In Phase 1 of this project, we investigated enhancing MatchFormer's geometric reasoning through epipolar constraints:

1. **Epipolar masking at inference**: We applied a soft Gaussian mask to the coarse confidence matrix after dual-softmax normalization, weighting matches by their distance to the epipolar line. This improved pose estimation AUC@20 from 0.00 to 8.48 on 323 hard ScanNet pairs where the vanilla model fails — a 39% recovery rate on previously unsolvable cases.

2. **Benchmarking on WxBS**: We evaluated epipolar masking across 8 challenging outdoor scenes from the WxBS dataset, comparing Laplacian, Gaussian, and hard decay functions. Gaussian masking with sigma=10 emerged as the best default, reducing mean reprojection error by up to 14x on hard scenes (e.g., Strahov: 114px to 8px).

3. **Fine-tuning infrastructure**: We built a complete fine-tuning pipeline for MatchFormer including:
   - Frozen backbone training (Block1/2 frozen, Block3/4 + FPN head trainable, all BatchNorm frozen)
   - Per-pair epipolar mask computation with Gaussian decay
   - Multi-scene ScanNet training with deterministic per-scene train/test splits
   - Focal loss with configurable hard negative sampling
   - Comprehensive benchmarking scripts for pose AUC and matching precision

4. **Key findings**:
   - Epipolar masking at inference requires known camera intrinsics and extrinsics — limiting its applicability
   - Fine-tuning with standard correspondence losses (focal loss on GT depth-projected matches) struggles to improve over the pretrained baseline
   - BatchNorm corruption from small-batch fine-tuning is a critical failure mode that must be addressed by freezing BN statistics
   - Loss collapse occurs when training on easy pairs (small frame gap); harder pairs and hard negative sampling are essential

**The fundamental limitation**: Epipolar masking works well but requires calibration at test time. A model that has *internalized* geometric reasoning — producing epipolar-consistent matches without any mask — would be far more useful in practice.

---

## 2. Proposed Work

### 2.1 Core Idea

We propose to fine-tune MatchFormer (and optionally LoFTR) to produce geometrically robust matches on challenging OOD datasets, using **only camera intrinsics and extrinsics as supervision** — no ground-truth correspondences required.

This draws direct motivation from **SCENES** (Kloepfer et al., 2024), which demonstrated that epipolar geometry alone provides sufficient supervision for adapting feature matchers to new domains. Their key insight: rather than requiring expensive 3D structure (depth maps, point clouds, or manually annotated correspondences), the epipolar constraint — that corresponding points must lie on each other's epipolar lines — provides a free geometric training signal derivable from camera poses alone.

### 2.2 Methodology

#### Epipolar Curriculum Training (Three Phases)

We adopt a phased curriculum that progressively transfers geometric reasoning from an external scaffold (the epipolar mask) into the network's learned weights:

**Phase 1 — Scaffolded Training (~10k steps)**
- Full-strength Gaussian epipolar mask applied after dual-softmax on the coarse confidence matrix
- Standard focal loss + fine L2 loss (same as baseline MatchFormer training)
- The mask constrains the matching search space, forcing cross-attention to focus on epipolar-consistent regions
- No epipolar loss term; the mask is the sole geometric signal

**Phase 2 — Annealing + Loss Handoff (~15k steps)**
- The mask bandwidth grows exponentially (sigma increases), progressively relaxing the geometric constraint
- Simultaneously, a SCENES-style **Sampson epipolar loss** ramps up:
  ```
  L_epi = (1/N) * sum( (p1^T F p0)^2 / (||Fp0||^2 + ||F^T p1||^2) )
  ```
  This penalizes predicted matches that violate the epipolar constraint, using only the fundamental matrix F (derived from poses and intrinsics)
- The mask catches the network if it drifts geometrically; the loss teaches it to self-correct

**Phase 3 — Free Flight (~10k steps)**
- Mask completely removed; the network matches purely on learned features
- Light residual epipolar loss (10% of peak weight) acts as geometric regularizer
- Mixed frame gaps (20-50) force the model to handle varying difficulty without assistance

#### Domain Adaptation via Epipolar-Only Supervision

For adapting to OOD datasets (EuRoC-MAV, WxBS), we follow SCENES' approach:

1. **No GT correspondences needed**: We use only the camera trajectory (poses) and intrinsics from the target domain. For EuRoC-MAV, these come from the Vicon motion capture ground truth. For WxBS, we derive pseudo-GT poses from known correspondences via essential matrix decomposition.

2. **Bootstrapping with uncertain poses**: Following SCENES, when only approximate poses are available (e.g., from visual odometry), we use a bootstrapping loop:
   - Fine-tune with current pose estimates
   - Use the improved matcher to re-estimate poses
   - Repeat until convergence

3. **Domain-specific challenges**:
   - **EuRoC-MAV**: Fast drone motion causes motion blur, large viewpoint changes between frames, and aggressive rotations. We address this with wider frame gaps and blur-aware data augmentation.
   - **WxBS**: Extreme illumination changes (day/night), large scale differences, and repeated textures. We focus on learning illumination-invariant features through the cross-attention adaptation.

### 2.3 Key Training Details

| Parameter | Phase 1 | Phase 2 | Phase 3 |
|-----------|---------|---------|---------|
| Mask strength | Full (sigma_0=3 coarse px) | Annealing (sigma grows) | Off |
| Epipolar loss weight | 0 | 0 -> lambda_max | lambda_max -> 0.1*lambda_max |
| Learning rate | 3e-5 (cosine) | 2e-5 (cosine) | 1e-5 (cosine) |
| Frame gap | 40 (constant) | 20 -> 40 (increasing) | 20-50 (mixed) |
| Frozen layers | Block1/2, all BN | Block1/2, all BN | Block1/2, all BN |

### 2.4 What Makes This Different from SCENES

While inspired by SCENES, our approach differs in several ways:

1. **Architecture**: We adapt MatchFormer's interleaved attention architecture (not SuperGlue/LoFTR), which integrates feature extraction and matching in a single backbone.
2. **Curriculum**: SCENES applies epipolar loss from step 0. We use a three-phase curriculum that first scaffolds with a geometric mask, then transitions to the loss. This is gentler and should preserve more pretrained knowledge.
3. **Selective freezing**: We freeze the low-level vision layers (Block1/2) and all BatchNorm statistics, training only the high-level cross-attention and FPN matching layers.
4. **Hard negative sampling**: We sample on-epipolar hard negatives — points along the correct epipolar line but at wrong depths — which SCENES does not explicitly address.

---

## 3. Evaluation Plan

### 3.1 Datasets

| Dataset | Domain | Challenges | Pose Source |
|---------|--------|-----------|-------------|
| **ScanNet** (train) | Indoor rooms | Baseline training domain | GT poses |
| **EuRoC-MAV** | Indoor drone | Motion blur, fast rotation, aggressive trajectories | Vicon GT |
| **WxBS** | Outdoor landmarks | Extreme illumination, scale, viewpoint changes | Pseudo-GT from correspondences |
| **ScanNet held-out** | Indoor rooms | In-distribution test | GT poses |

### 3.2 Metrics

1. **Pose Estimation AUC**: AUC at angular error thresholds @5, @10, @20 degrees — the primary metric for geometric quality
2. **Matching Precision**: Percentage of predicted matches within 3px and 5px of ground truth
3. **Match Count**: Number of confident matches per pair (with confidence > threshold)
4. **Geometric Internalization Ratio (GIR)**: Ratio of mask-off to mask-on AUC@5 — measures how much geometric reasoning the network has internalized

### 3.3 Baselines

| Condition | Description |
|-----------|-------------|
| **Vanilla MatchFormer** | Pretrained model, no fine-tuning |
| **Vanilla + Epipolar Mask** | Inference-time masking (requires calibration) |
| **Correspondence-supervised** | Fine-tuned with GT depth-projected matches (standard approach) |
| **Epipolar-only (ours)** | Fine-tuned with epipolar curriculum (no GT correspondences) |

### 3.4 Ablation Study

| Ablation | Purpose |
|----------|---------|
| Mask-only (no annealing) | Isolate mask contribution vs. curriculum |
| Loss-only (no mask) | Compare SCENES-style direct loss vs. our curriculum |
| Mask + Loss (no annealing) | Full signal throughout — does curriculum matter? |
| Full curriculum | Complete three-phase approach |

---

## 4. Expected Contributions

1. **Correspondence-free fine-tuning**: Demonstrate that MatchFormer can be effectively adapted to new domains using only camera poses — no depth maps, point clouds, or manually annotated correspondences.

2. **Epipolar curriculum**: A principled three-phase training strategy that transfers geometric knowledge from an external mask into the network's weights, combining the speed of mask-based training with the generality of self-supervised geometric learning.

3. **OOD robustness**: Quantified improvements on EuRoC-MAV and WxBS over the pretrained baseline, showing that domain-specific geometric adaptation is possible without domain-specific ground truth.

4. **Practical impact**: A fine-tuned model that deploys without requiring camera calibration at inference — unlike the epipolar masking approach which needs known poses and intrinsics.

---

## 5. Timeline

| Week | Milestone |
|------|-----------|
| 1 | Complete Phase 1 training on ScanNet (6 scenes), validate GIR diagnostic |
| 2 | Implement Sampson epipolar loss, run Phase 2 annealing on ScanNet |
| 3 | Phase 3 free flight + full ScanNet benchmark (in-distribution) |
| 4 | Adapt pipeline to EuRoC-MAV, run epipolar-only fine-tuning |
| 5 | Adapt pipeline to WxBS, run epipolar-only fine-tuning |
| 6 | Ablation study (4 conditions), final benchmarks, paper writeup |

---

## 6. References

1. Kloepfer, D. A., Henriques, J. F., & Campbell, D. (2024). SCENES: Subpixel Correspondence Estimation with Epipolar Supervision. *arXiv preprint arXiv:2401.10886*.
2. Wang, Q., et al. (2022). MatchFormer: Interleaving Attention in Transformers for Feature Matching. *ACCV 2022*.
3. Sun, J., et al. (2021). LoFTR: Detector-Free Local Feature Matching with Transformers. *CVPR 2021*.
4. Burri, M., et al. (2016). The EuRoC micro aerial vehicle datasets. *IJRR*.
5. Mishkin, D., et al. (2015). WxBS: Wide Baseline Stereo Generalizations. *BMVC*.
