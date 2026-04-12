"""
Fine-tuning launcher for self-supervised epipolar MatchFormer.

The loss pipeline lives entirely in:
  - model/supervision.py     → computes F_list + epi_mask from poses/intrinsics
  - model/losses.py          → epi_focal_loss + soft_sampson_loss → MatchFormerLoss
  - model/lightning_loftr.py → PL_LoFTR.training_step calls supervision + criterion

This script just wires CLI args into config and launches the trainer.

Usage:
    # Overfit sanity check
    python train_finetune.py --overfit

    # Full fine-tune
    python train_finetune.py --steps 10000 --lambda_epi 0.7 --sampson_margin 1.0

    # Resume
    python train_finetune.py --resume checkpoints/last.ckpt --steps 10000
"""

import os
import re
import glob
import sys
import argparse
import numpy as np
import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config.defaultmf import get_cfg_defaults
from model.lightning_loftr import PL_LoFTR
from model.datasets.scannet_simple import ScanNetSimpleDataset
from model.utils.metrics import estimate_pose, relative_pose_error, error_auc


# ── Collate function ─────────────────────────────────────────────────────────

def collate_fn(batch):
    """Collate a list of dataset items into a batch dict."""
    keys = batch[0].keys()
    out = {}
    for k in keys:
        if k in ['pair_names', 'hw0_i', 'hw0_c']:
            out[k] = [b[k] for b in batch]
            if k in ['hw0_i', 'hw0_c']:
                out[k] = out[k][0]  # all images in batch share resize size
        else:
            out[k] = torch.stack([b[k] for b in batch])
    return out


# ── LR override callback (Phase 2: resume Phase 1 ckpt at lower lr) ──────────

class LROverrideCallback(pl.Callback):
    def __init__(self, lr):
        self.lr = lr

    def on_train_start(self, trainer, pl_module):
        for optimizer in trainer.optimizers:
            for param_group in optimizer.param_groups:
                param_group['lr'] = self.lr
        print(f"[LROverride] Learning rate set to {self.lr}")


class PoseAUCCallback(pl.Callback):
    """
    Every `every_n_steps` training steps, runs pose estimation over the val
    dataset and logs AUC@5/10/20, precision, and mean num_matches to W&B.

    Mirrors benchmark_test_split.py but runs on the in-memory fine-tuned model,
    with configurable confidence thresholds (default [0.2, 0.05, 0.01]).
    """
    def __init__(self, val_loader, every_n_steps=1000, thresholds=(0.2, 0.05, 0.01),
                 ransac_thresh=0.5, ransac_conf=0.99999):
        self.val_loader = val_loader
        self.every_n_steps = every_n_steps
        self.thresholds = sorted(thresholds)  # ascending so lowest is fwd thr
        self.ransac_thresh = ransac_thresh
        self.ransac_conf = ransac_conf

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        step = trainer.global_step
        if step == 0 or step % self.every_n_steps != 0:
            return
        self._run_benchmark(trainer, pl_module, step)

    @torch.no_grad()
    def _run_benchmark(self, trainer, pl_module, step):
        device = pl_module.device
        matcher = pl_module.matcher
        was_training = matcher.training
        matcher.eval()

        # Forward once at the lowest threshold, post-filter by mconf for the rest
        fwd_thr = self.thresholds[0]
        orig_thr = matcher.coarse_matching.thr
        matcher.coarse_matching.thr = fwd_thr

        results = {thr: {'R_errs': [], 't_errs': [], 'precisions': [], 'n_matches': []}
                   for thr in self.thresholds}

        for batch in self.val_loader:
            T0 = batch['T0'][0].numpy()
            T1 = batch['T1'][0].numpy()
            K = batch['K'][0].numpy()
            if not (np.isfinite(T0).all() and np.isfinite(T1).all()):
                continue
            T_0to1 = np.linalg.inv(T1) @ T0

            input_data = {
                'image0': batch['image0'].to(device, non_blocking=True),
                'image1': batch['image1'].to(device, non_blocking=True),
            }
            with torch.inference_mode():
                matcher(input_data)

            mkpts0 = input_data['mkpts0_f'].cpu().numpy()
            mkpts1 = input_data['mkpts1_f'].cpu().numpy()
            mconf = input_data.get('mconf')
            mconf = mconf.cpu().numpy() if mconf is not None else np.ones(len(mkpts0))

            for thr in self.thresholds:
                if thr > fwd_thr:
                    keep = mconf >= thr
                    mk0, mk1 = mkpts0[keep], mkpts1[keep]
                else:
                    mk0, mk1 = mkpts0, mkpts1

                n = len(mk0)
                ret = estimate_pose(mk0, mk1, K, K,
                                    thresh=self.ransac_thresh, conf=self.ransac_conf)
                if ret is None:
                    results[thr]['R_errs'].append(np.inf)
                    results[thr]['t_errs'].append(np.inf)
                    results[thr]['precisions'].append(0.0)
                else:
                    R, t, inliers = ret
                    t_err, R_err = relative_pose_error(T_0to1, R, t, ignore_gt_t_thr=0.0)
                    results[thr]['R_errs'].append(R_err)
                    results[thr]['t_errs'].append(t_err)
                    results[thr]['precisions'].append(
                        float(np.mean(inliers)) if len(inliers) > 0 else 0.0)
                results[thr]['n_matches'].append(n)

        # Log AUC + precision + num_matches for each threshold
        for thr in self.thresholds:
            r = results[thr]
            if len(r['R_errs']) == 0:
                continue
            pose_errors = np.maximum(np.array(r['R_errs']), np.array(r['t_errs']))
            aucs = error_auc(pose_errors, [5, 10, 20])
            tag = f"bench_thr{thr:.2f}"
            pl_module.log(f'{tag}/auc5',  float(aucs['auc@5']  * 100), on_step=True)
            pl_module.log(f'{tag}/auc10', float(aucs['auc@10'] * 100), on_step=True)
            pl_module.log(f'{tag}/auc20', float(aucs['auc@20'] * 100), on_step=True)
            pl_module.log(f'{tag}/precision',  float(np.mean(r['precisions']) * 100), on_step=True)
            pl_module.log(f'{tag}/num_matches', float(np.mean(r['n_matches'])), on_step=True)

        print(f"\n[PoseAUC step={step}] " + " | ".join(
            f"thr={thr:.2f} AUC@5/10/20={error_auc(np.maximum(np.array(results[thr]['R_errs']), np.array(results[thr]['t_errs'])), [5,10,20])['auc@5']*100:.1f}"
            f"/{error_auc(np.maximum(np.array(results[thr]['R_errs']), np.array(results[thr]['t_errs'])), [5,10,20])['auc@10']*100:.1f}"
            f"/{error_auc(np.maximum(np.array(results[thr]['R_errs']), np.array(results[thr]['t_errs'])), [5,10,20])['auc@20']*100:.1f} "
            f"n={np.mean(results[thr]['n_matches']):.0f}"
            for thr in self.thresholds if len(results[thr]['R_errs']) > 0))

        # Restore
        matcher.coarse_matching.thr = orig_thr
        if was_training:
            matcher.train()


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir',       default='../data/scans')
    parser.add_argument('--ckpt',           default='model/weights/indoor-lite-LA.ckpt')
    parser.add_argument('--steps',          type=int, default=10000)
    parser.add_argument('--lr',             type=float, default=1e-4)
    parser.add_argument('--batch',          type=int, default=2)
    parser.add_argument('--workers',        type=int, default=2)
    parser.add_argument('--frame_gap',      type=int, default=20)
    parser.add_argument('--random_gap',     type=str, default=None,
                        help='Random gap range "min,max" (overrides --frame_gap).')
    parser.add_argument('--resume',         default=None)
    parser.add_argument('--checkpoint_dir', default='checkpoints/')
    parser.add_argument('--save_every',     type=int, default=500)
    parser.add_argument('--overfit',        action='store_true')
    parser.add_argument('--precision',      default='32')
    parser.add_argument('--neg_per_pos',    type=int, default=0)
    parser.add_argument('--no_freeze',      action='store_true')
    parser.add_argument('--override_lr',    action='store_true')
    parser.add_argument('--wandb',          action='store_true')
    parser.add_argument('--wandb_project',  default='matchformer-finetune')
    parser.add_argument('--wandb_run',      default=None)
    parser.add_argument('--split_seed',     type=int, default=42)
    parser.add_argument('--scenes',         nargs='+', default=None)
    parser.add_argument('--split_ratio',    type=float, default=0.9)
    parser.add_argument('--eta_min',        type=float, default=1e-6)
    parser.add_argument('--split_mode',     default='sequential',
                        choices=['sequential', 'random'])
    parser.add_argument('--lambda_c',       type=float, default=1.0,
                        help='Weight on coarse loss.')
    parser.add_argument('--lambda_f',       type=float, default=0.0,
                        help='Weight on fine loss. 0 in pose-only setup '
                             '(fine loss needs depth-derived GT).')
    parser.add_argument('--lambda_epi',     type=float, default=0.7,
                        help='L_coarse = (1-λ)·L_focal_epi + λ·L_sampson. Default 0.7.')
    parser.add_argument('--sampson_margin', type=float, default=1.0,
                        help='Geometric dead zone in pixels for Sampson loss.')
    parser.add_argument('--epi_thresh',     type=float, default=2.0,
                        help='Pixel distance to epipolar line for binary epi_mask.')
    parser.add_argument('--bench_every',    type=int, default=1000,
                        help='Run pose-AUC benchmark on val split every N steps. '
                             '0 = disabled.')
    parser.add_argument('--bench_thresholds', type=float, nargs='+',
                        default=[0.1, 0.05, 0.01],
                        help='Confidence thresholds for pose-AUC benchmark.')
    args = parser.parse_args()

    os.makedirs(args.checkpoint_dir, exist_ok=True)

    # ── Auto-resume: prefer highest epipolar-N.ckpt, fall back to last.ckpt ───
    resume_path = args.resume
    if resume_path is None:
        step_ckpts = [p for p in glob.glob(os.path.join(args.checkpoint_dir, 'epipolar-*.ckpt'))
                      if re.search(r'epipolar-step=(\d+)\.ckpt', p)]
        if step_ckpts:
            resume_path = max(step_ckpts, key=lambda p: int(re.search(r'epipolar-step=(\d+)', p).group(1)))
            print(f"[Auto-resume] Resuming from: {resume_path}")
        else:
            last_ckpt = os.path.join(args.checkpoint_dir, 'last.ckpt')
            if os.path.exists(last_ckpt):
                resume_path = last_ckpt
                print(f"[Auto-resume] Resuming from: {resume_path}")
            else:
                print("[Auto-resume] No checkpoint found — starting from scratch.")

    # ── Config ──────────────────────────────────────────────────────────────
    config = get_cfg_defaults()
    config.MATCHFORMER.BACKBONE_TYPE = 'litela'
    config.MATCHFORMER.SCENS = 'indoor'
    config.MATCHFORMER.RESOLUTION = (8, 4)
    config.MATCHFORMER.COARSE.D_MODEL = 192
    config.MATCHFORMER.COARSE.D_FFN = 192
    config.LR             = args.lr
    config.TOTAL_STEPS    = args.steps
    config.LOSS_LAMBDA_C  = args.lambda_c
    config.LOSS_LAMBDA_F  = args.lambda_f
    config.NEG_PER_POS    = args.neg_per_pos
    config.ETA_MIN        = args.eta_min
    config.LAMBDA_EPI     = args.lambda_epi
    config.SAMPSON_MARGIN = args.sampson_margin
    config.EPI_THRESH     = args.epi_thresh

    # ── Dataset ─────────────────────────────────────────────────────────────
    max_pairs = 5 if args.overfit else None
    random_gap_range = None
    if args.random_gap:
        parts = args.random_gap.split(',')
        random_gap_range = (int(parts[0]), int(parts[1]))
        print(f'Random gap range: {random_gap_range}')

    if args.overfit:
        dataset = ScanNetSimpleDataset(
            args.data_dir, frame_gap=args.frame_gap, max_pairs=max_pairs,
            random_gap_range=random_gap_range, scenes=args.scenes,
        )
        train_ds, val_ds = dataset, dataset
        print(f"Overfit mode: {len(dataset)} pairs")
    else:
        train_ds = ScanNetSimpleDataset(
            args.data_dir, frame_gap=args.frame_gap, max_pairs=max_pairs,
            random_gap_range=random_gap_range, scenes=args.scenes,
            split='train', split_ratio=args.split_ratio,
            split_mode=args.split_mode, split_seed=args.split_seed,
        )
        val_ds = ScanNetSimpleDataset(
            args.data_dir, frame_gap=args.frame_gap, max_pairs=None,
            random_gap_range=random_gap_range, scenes=args.scenes,
            split='test', split_ratio=args.split_ratio,
            split_mode=args.split_mode, split_seed=args.split_seed,
        )
        print(f"Train: {len(train_ds)} | Val: {len(val_ds)} "
              f"(per-scene {args.split_ratio:.0%}/{1-args.split_ratio:.0%} split)")

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                              num_workers=args.workers, collate_fn=collate_fn,
                              pin_memory=True, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=1, shuffle=False,
                              num_workers=0, collate_fn=collate_fn)

    # ── Model: plain PL_LoFTR — supervision + losses are handled inside ─────
    model = PL_LoFTR(config, pretrained_ckpt=args.ckpt,
                     freeze_backbone=not args.no_freeze)

    # ── Callbacks & Trainer ──────────────────────────────────────────────────
    callbacks = [
        pl.callbacks.ModelCheckpoint(
            dirpath=args.checkpoint_dir,
            filename='epipolar-{step:05d}',
            every_n_train_steps=args.save_every,
            save_last=True,
            save_top_k=-1,
        ),
        pl.callbacks.LearningRateMonitor(logging_interval='step'),
    ]
    if args.override_lr:
        callbacks.append(LROverrideCallback(args.lr))
        print(f"[Phase 2] LR override active: will set lr={args.lr} after checkpoint load")

    if args.bench_every > 0 and not args.overfit:
        callbacks.append(PoseAUCCallback(
            val_loader=val_loader,
            every_n_steps=args.bench_every,
            thresholds=tuple(args.bench_thresholds),
        ))
        print(f"[PoseAUC] Benchmarking every {args.bench_every} steps "
              f"at thresholds {args.bench_thresholds}")

    precision = args.precision
    if precision == 'bf16':
        precision = 'bf16-mixed'

    # ── W&B logger (optional) ────────────────────────────────────────────────
    loggers = []
    if args.wandb:
        run_name = args.wandb_run or (
            f"phase2-lr{args.lr}-neg{args.neg_per_pos}" if args.override_lr
            else f"phase1-lr{args.lr}"
        )
        wandb_logger = WandbLogger(
            project=args.wandb_project,
            name=run_name,
            config={
                'lr': args.lr,
                'steps': args.steps,
                'batch': args.batch,
                'neg_per_pos': args.neg_per_pos,
                'frame_gap': args.frame_gap,
                'random_gap': args.random_gap,
                'lambda_c': args.lambda_c,
                'lambda_f': args.lambda_f,
                'lambda_epi': args.lambda_epi,
                'sampson_margin': args.sampson_margin,
                'epi_thresh': args.epi_thresh,
                'precision': precision,
                'resume': resume_path,
            },
        )
        loggers.append(wandb_logger)
        print(f"[W&B] Logging to project='{args.wandb_project}' run='{run_name}'")

    trainer = pl.Trainer(
        max_steps=args.steps,
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=1,
        precision=precision,
        log_every_n_steps=1 if args.overfit else 10,
        limit_val_batches=0.0 if args.overfit else 1.0,
        val_check_interval=1.0,
        callbacks=callbacks,
        logger=loggers if loggers else True,
        enable_progress_bar=True,
        gradient_clip_val=1.0,
    )

    trainer.fit(model, train_loader, val_loader, ckpt_path=resume_path)
    print(f"Training complete. Checkpoints saved to {args.checkpoint_dir}")


if __name__ == '__main__':
    main()
