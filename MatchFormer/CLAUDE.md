# MatchFormer — Epipolar Fine-Tuning Project

## Quick Reference

- **Python env**: `source /Users/siddharthraj/classes/cv/cv_final/myenv/bin/activate`
- **Data**: `../data/scans/` contains ScanNet scenes (e.g. `scene0000_00/exported/`)
- **Pretrained weights**: `model/weights/indoor-lite-LA.ckpt`
- **Architecture**: `litela` variant — `BACKBONE_TYPE=litela`, `D_MODEL=192`, `D_FFN=192`, `RESOLUTION=(8,4)`

## Project Structure

```
MatchFormer/
├── model/
│   ├── matchformer.py          # Main model: backbone → coarse → fine_preprocess → fine_matching
│   ├── lightning_loftr.py      # PL training module (PL_LoFTR, EpipolarFineTuner lives in train_finetune.py)
│   ├── supervision.py          # GT coarse correspondences from depth+pose projection
│   ├── losses.py               # FocalLoss (coarse) + fine_loss (L2 on fine offset)
│   ├── backbone/
│   │   ├── match_LA_lite.py    # litela backbone (AttentionBlock1-4 + FPN)
│   │   ├── coarse_matching.py  # Dual-softmax confidence matrix → mutual NN → matches
│   │   ├── fine_matching.py    # Sub-pixel refinement via spatial expectation (no learnable params)
│   │   └── fine_preprocess.py  # Unfold fine patches around coarse matches
│   ├── datasets/
│   │   └── scannet_simple.py   # ScanNetSimpleDataset for fine-tuning
│   └── weights/                # Checkpoints
├── train_finetune.py           # Fine-tuning with epipolar constraint
├── benchmark_test_split.py     # Pose AUC benchmark on 10% val split
├── benchmark_distance_test_split.py  # Reprojection distance benchmark on 10% val split
├── pose_benchmark.py           # Multi-scene Table 2 style benchmark
├── gt_epipolar.py              # Fundamental matrix computation utility
├── config/defaultmf.py         # Default config (yacs)
├── temp/                       # Temporary/verification files (GT projection PNGs, etc.)
├── reports-txt/                # Benchmark log files (.txt)
├── reports-md/                 # Benchmark/analysis reports (.md)
└── reports/                    # Final report (PDF, tex, figures)
```

## Training

```bash
# Default: freeze backbone, train AttentionBlock3/4 + FPN head only
python train_finetune.py --data_dir ../data/scans --steps 50000 --tau 10 --wandb

# NEVER use --no_freeze unless you have a very large dataset.
# Unfreezing all layers on small data causes catastrophic forgetting.
```

Key args: `--lr` (1e-4), `--tau` (10.0), `--batch` (2), `--frame_gap` (20), `--split_seed` (42), `--neg_per_pos` (0, use 15 for multi-scene phase 2), `--save_every` (500).

Train/val split: 90/10 via `torch.random_split` with `Generator().manual_seed(split_seed)`. Use the same seed locally and on Colab to get identical splits.

## Benchmarking

**Do NOT re-run Vanilla and Vanilla+Epipolar baselines every time.** Their results don't change. Only run the new fine-tuned model the user asks about. Re-run baselines only when the data/scene/split changes or the user explicitly asks.

```bash
# Pose AUC (Table 2 style) on 10% val split of one scene
python benchmark_test_split.py --data_dir ../data/scans/scene0000_00/exported --finetuned_ckpt model/weights/MY_CKPT.ckpt

# Reprojection distance on 10% val split
python benchmark_distance_test_split.py --data_dir ../data/scans/scene0000_00/exported --finetuned_ckpt model/weights/MY_CKPT.ckpt

# Multi-scene pose AUC (OOD scenes 11-15)
python pose_benchmark.py --scenes scene0011_00 scene0012_00 scene0013_00 scene0014_00 scene0015_00
```

## Architecture Notes

MatchFormer interleaves feature extraction and matching in a single backbone (unlike LoFTR which has separate CNN + transformer):

| Block | Resolution | Attention Pattern | Role |
|-------|-----------|-------------------|------|
| AttentionBlock1 | 1/4 | Self, Self, Cross | Low-level features + early matching |
| AttentionBlock2 | 1/8 | Self, Self, Cross | Mid-level features |
| **AttentionBlock3** | 1/16 | Self, **Cross, Cross** | Heavy cross-attention — main matching |
| **AttentionBlock4** | 1/32 | Self, **Cross, Cross** | Global matching reasoning |

Block3/4 outputs feed into FPN → coarse features (1/8) and fine features (1/4).

**Frozen training** unfreezes: AttentionBlock3, AttentionBlock4, layer1_outconv, layer1_outconv2. This preserves pretrained visual features (Block1/2) while adapting the matching layers.

## Critical Conventions

- **No T_cv2gl**: ScanNet poses are already in OpenCV convention. Do NOT apply any OpenGL-to-OpenCV transform. This was a previous bug that corrupted training.
- **Poses are camera-to-world** (T_cw). Relative transform: `T_0to1 = inv(T1) @ T0`.
- **Depth**: stored in mm in PNG files, converted to metres (`/ 1000.0`). Valid range: 0.1–10.0m.
- **Image size**: always resized to 640x480.
- **Coarse stride**: 8 (feature map is 80x60). Fine stride: 4 (feature map is 160x120).
- **Confidence threshold**: vanilla uses 0.2. Fine-tuned models may need lower thresholds (0.01–0.05).

## Dataset Format

```
scene_dir/exported/
├── color/      # 0.jpg, 1.jpg, ...
├── depth/      # 0.png, 1.png, ... (uint16, millimetres)
├── pose/       # 0.txt, 1.txt, ... (4x4 camera-to-world)
└── intrinsic/  # intrinsic_depth.txt (4x4, use top-left 3x3)
```

## Loss

- **Coarse**: Focal loss on confidence matrix (lambda_c=1.0). GT from depth-based reprojection in `supervision.py`.
- **Fine**: L2 loss on predicted sub-pixel offset vs (0,0) center, weighted by 1/std (lambda_f=0.5). `loss_f ≈ 0` is normal — pretrained fine matching already works well.
