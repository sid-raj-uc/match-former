"""
Fine-tuning launcher for epipolar-constrained MatchFormer.

Trains the model with:
  - Epipolar mask baked into CoarseMatching forward pass (inference-time)
  - GT coarse correspondences from depth projection (supervision)
  - Focal loss on confidence matrix + L2 on fine-level offsets

Usage:
    # Overfitting sanity check (fast, on 5 pairs, should converge in ~100 steps)
    python train_finetune.py --overfit

    # Full fine-tune
    python train_finetune.py --steps 10000
    
    # Resume from checkpoint
    python train_finetune.py --resume checkpoints/last.ckpt --steps 10000
"""

import os
import re
import glob
import sys
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader, random_split

# ── Project imports ─────────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config.defaultmf import get_cfg_defaults
from model.lightning_loftr import PL_LoFTR
from model.datasets.scannet_simple import ScanNetSimpleDataset
from model.backbone.coarse_matching import CoarseMatching
from model.losses import sampson_epipolar_loss, epipolar_coarse_loss
from gt_epipolar import compute_fundamental_matrix

# ── Epipolar Constraint Setup ─────────────────────────────────────────────────

def get_epipolar_mask_matrix(F_mat, H0, W0, H1, W1, H_img=480, W_img=640, tau=10.0, device='cpu'):
    y0, x0 = torch.meshgrid(torch.arange(H0), torch.arange(W0), indexing='ij')
    x0_img = (x0.float() / W0) * W_img
    y0_img = (y0.float() / H0) * H_img
    y1, x1 = torch.meshgrid(torch.arange(H1), torch.arange(W1), indexing='ij')
    x1_img = (x1.float() / W1) * W_img
    y1_img = (y1.float() / H1) * H_img
    
    p0 = torch.stack([x0_img.flatten(), y0_img.flatten(), torch.ones(H0*W0)], dim=1).to(device)
    p1 = torch.stack([x1_img.flatten(), y1_img.flatten(), torch.ones(H1*W1)], dim=1).to(device)
    F_t = torch.tensor(F_mat, dtype=torch.float32, device=device)
    l_prime = p0 @ F_t.T  # [N0, 3]
    denom = torch.sqrt(l_prime[:, 0]**2 + l_prime[:, 1]**2).unsqueeze(1)  # [N0, 1]

    # Chunk to avoid OOM: process CHUNK_SIZE rows of l_prime at a time
    CHUNK_SIZE = 512
    N0 = l_prime.shape[0]
    mask_rows = []
    for start in range(0, N0, CHUNK_SIZE):
        end = min(start + CHUNK_SIZE, N0)
        l_chunk = l_prime[start:end]          # [chunk, 3]
        d_chunk = denom[start:end]            # [chunk, 1]
        num_chunk = torch.abs(l_chunk @ p1.T) # [chunk, N1]
        dist_chunk = num_chunk / (d_chunk + 1e-8)
        mask_rows.append(torch.exp(-dist_chunk**2 / (2 * tau**2)))
    mask = torch.cat(mask_rows, dim=0)  # [N0, N1]
    return mask.unsqueeze(0)  # [1, N0, N1]


original_coarse_forward = CoarseMatching.forward

def epipolar_coarse_forward(self, feat_c0, feat_c1, data, mask_c0=None, mask_c1=None):
    N, L, S, C = feat_c0.size(0), feat_c0.size(1), feat_c1.size(1), feat_c0.size(2)
    feat_c0_n = feat_c0 / feat_c0.shape[-1]**0.5
    feat_c1_n = feat_c1 / feat_c1.shape[-1]**0.5
    sim_matrix = torch.einsum("nlc,nsc->nls", feat_c0_n, feat_c1_n) / self.temperature
    if mask_c0 is not None:
        sim_matrix.masked_fill_(~(mask_c0[..., None] * mask_c1[:, None]).bool(), -1e9)
    conf_matrix = F.softmax(sim_matrix, 1) * F.softmax(sim_matrix, 2)

    data.update({'conf_matrix': conf_matrix, 'sim_matrix': sim_matrix})
    data.update(**self.get_coarse_match(conf_matrix, data))

CoarseMatching.forward = epipolar_coarse_forward


# ── Collate function ─────────────────────────────────────────────────────────

def collate_fn(batch):
    """Collate a list of dataset items into a batch dict."""
    keys = batch[0].keys()
    out = {}
    for k in keys:
        if k in ['pair_names', 'hw0_i', 'hw0_c']:
            out[k] = [b[k] for b in batch]
            if k in ['hw0_i', 'hw0_c']:
                out[k] = out[k][0] # Just take the first one since all images in a batch are resized to the same dimensions
        else:
            out[k] = torch.stack([b[k] for b in batch])
    return out


# ── Training with per-batch epipolar F injection ──────────────────────────────

class LROverrideCallback(pl.Callback):
    """
    Overrides the optimizer learning rate after a checkpoint is restored.
    Without this, resuming a checkpoint replays the saved optimizer state
    (including its LR), ignoring the --lr argument. This callback fires
    at the very start of training and forces the LR to the requested value.
    Use for phase 2: resume from phase 1 checkpoint at a lower LR.
    """
    def __init__(self, lr):
        self.lr = lr

    def on_train_start(self, trainer, pl_module):
        for optimizer in trainer.optimizers:
            for param_group in optimizer.param_groups:
                param_group['lr'] = self.lr
        print(f"[LROverride] Learning rate set to {self.lr}")


class EpipolarFineTuner(PL_LoFTR):
    """
    Extends PL_LoFTR with per-batch epipolar F matrix computation.
    Before each forward pass in training, we compute F from the batch poses
    and inject it into CoarseMatching.
    """
    def __init__(self, config, pretrained_ckpt=None, tau=10.0, freeze_backbone=True, lambda_epi=0.0):
        super().__init__(config, pretrained_ckpt=pretrained_ckpt, freeze_backbone=freeze_backbone)
        self.tau = tau
        self.lambda_epi = lambda_epi
        self._current_F_list = None  # stored per batch for loss computation

    def _inject_epipolar(self, batch):
        """Compute F for each pair in the batch and inject into coarse_matching."""
        B = batch['T0'].shape[0]
        F_list = []
        for i in range(B):
            T0 = batch['T0'][i].cpu().numpy()
            T1 = batch['T1'][i].cpu().numpy()
            K  = batch['K'][i].cpu().numpy()
            try:
                if not np.isfinite(T0).all() or not np.isfinite(T1).all():
                    F_list.append(None)
                    continue
                F_mat = compute_fundamental_matrix(T0, T1, K, K)
                if not np.isfinite(F_mat).all():
                    F_list.append(None)
                    continue
                F_list.append(F_mat)
            except Exception:
                F_list.append(None)
        self._current_F_list = F_list

    def _compute_epi_mask(self, batch):
        """Compute epipolar mask [B, L, S] from stored F matrices."""
        F_list = self._current_F_list
        if F_list is None:
            return None
        H0, W0 = batch['hw0_c']
        H1, W1 = batch['hw1_c']
        B = batch['image0'].shape[0]
        device = batch['image0'].device
        masks = []
        for i in range(B):
            if i < len(F_list) and F_list[i] is not None:
                m = get_epipolar_mask_matrix(F_list[i], H0, W0, H1, W1, tau=self.tau, device=device)
                masks.append(m.squeeze(0))
            else:
                masks.append(torch.ones(H0 * W0, H1 * W1, device=device))
        return torch.stack(masks, dim=0)

    def training_step(self, batch, batch_idx):
        self._inject_epipolar(batch)

        # Standard forward (populates conf_matrix, mkpts0_f, etc.)
        loss_standard = super().training_step(batch, batch_idx)

        if self.lambda_epi <= 0 or self._current_F_list is None:
            return loss_standard

        loss_total = loss_standard  # 0 on MH_05 (no depth), nonzero on ScanNet

        # --- SCENES coarse term: focal loss with epipolar mask as pseudo-GT ---
        conf_matrix = batch.get('conf_matrix')
        epi_mask = self._compute_epi_mask(batch)
        if conf_matrix is not None and epi_mask is not None:
            loss_coarse_epi = epipolar_coarse_loss(conf_matrix, epi_mask)
            loss_total = loss_total + (1 - self.lambda_epi) * loss_coarse_epi
            self.log('train/loss_coarse_epi', loss_coarse_epi.detach(), on_step=True, on_epoch=True)

        # --- SCENES fine term: Sampson distance on predicted matches ---
        mkpts0 = batch.get('mkpts0_f')
        mkpts1 = batch.get('mkpts1_f')
        m_bids = batch.get('m_bids')
        if mkpts0 is not None and len(mkpts0) > 0:
            loss_sampson = sampson_epipolar_loss(mkpts0, mkpts1, self._current_F_list, m_bids)
            loss_total = loss_total + self.lambda_epi * loss_sampson
            self.log('train/loss_sampson', loss_sampson.detach(), on_step=True, on_epoch=True)

        # Overwrite train/loss with the actual total (super logged only standard part)
        self.log('train/loss', loss_total.detach(), on_step=True, on_epoch=True, prog_bar=True)
        return loss_total

    def validation_step(self, batch, batch_idx):
        self._inject_epipolar(batch)
        loss_standard = super().validation_step(batch, batch_idx)

        if self.lambda_epi <= 0 or self._current_F_list is None:
            return loss_standard

        loss_total = loss_standard

        conf_matrix = batch.get('conf_matrix')
        epi_mask = self._compute_epi_mask(batch)
        if conf_matrix is not None and epi_mask is not None:
            loss_coarse_epi = epipolar_coarse_loss(conf_matrix, epi_mask)
            loss_total = loss_total + (1 - self.lambda_epi) * loss_coarse_epi
            self.log('val/loss_coarse_epi', loss_coarse_epi.detach(), on_step=False, on_epoch=True)

        mkpts0 = batch.get('mkpts0_f')
        mkpts1 = batch.get('mkpts1_f')
        m_bids = batch.get('m_bids')
        if mkpts0 is not None and len(mkpts0) > 0:
            loss_sampson = sampson_epipolar_loss(mkpts0, mkpts1, self._current_F_list, m_bids)
            loss_total = loss_total + self.lambda_epi * loss_sampson
            self.log('val/loss_sampson', loss_sampson.detach(), on_step=False, on_epoch=True)

        # Overwrite val/loss with the actual total (super logged only standard part)
        self.log('val/loss', loss_total.detach(), on_step=False, on_epoch=True, prog_bar=True)
        return loss_total


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='../data/scans', help='Root scans dir containing scene subdirs, or single exported scene dir.')
    parser.add_argument('--ckpt',           default='model/weights/indoor-lite-LA.ckpt')
    parser.add_argument('--steps',          type=int, default=10000)
    parser.add_argument('--lr',             type=float, default=1e-4)
    parser.add_argument('--tau',            type=float, default=10.0)
    parser.add_argument('--batch',          type=int, default=2)
    parser.add_argument('--workers',        type=int, default=2)
    parser.add_argument('--frame_gap',      type=int, default=20)
    parser.add_argument('--random_gap',     type=str, default=None,
                        help='Random gap range as "min,max" (e.g. "10,60"). '
                             'Overrides --frame_gap. Each pair samples a random gap per epoch.')
    parser.add_argument('--resume',         default=None,
                        help='Path to checkpoint to resume from. If not set, '
                             'auto-resumes from last.ckpt in --checkpoint_dir if it exists.')
    parser.add_argument('--checkpoint_dir', default='checkpoints/',
                        help='Directory to save checkpoints. Set to a Google Drive path on Colab.')
    parser.add_argument('--save_every',     type=int, default=500,
                        help='Save a checkpoint every N training steps.')
    parser.add_argument('--overfit',        action='store_true',
                        help='Overfit on 5 pairs to verify pipeline correctness')
    parser.add_argument('--precision',      default='32',
                        help='Training precision: 32, 16-mixed, or bf16 (L4/A100 recommended)')
    parser.add_argument('--neg_per_pos',    type=int, default=0,
                        help='Sampled negatives per positive in focal loss. '
                             '0 = use all negatives (original). '
                             '15 recommended for multi-scene phase 2 training.')
    parser.add_argument('--no_freeze',      action='store_true',
                        help='Train all weights (no freezing). Default freezes all but AttentionBlock3/4 and fine FPN head.')
    parser.add_argument('--override_lr',   action='store_true',
                        help='Override the learning rate stored in the resumed checkpoint. '
                             'Use this when resuming phase 1 checkpoint for phase 2 at a lower lr.')
    parser.add_argument('--wandb',          action='store_true',
                        help='Enable Weights & Biases logging.')
    parser.add_argument('--wandb_project',  default='matchformer-finetune',
                        help='W&B project name.')
    parser.add_argument('--wandb_run',      default=None,
                        help='W&B run name. Auto-generated if not set.')
    parser.add_argument('--split_seed',    type=int, default=42,
                        help='Random seed for train/val split. Use the same seed locally to reproduce the split.')
    parser.add_argument('--scenes',       nargs='+', default=None,
                        help='Scene names to train on (e.g. scene0000_00 scene0001_00). '
                             'If not set, uses all scenes in data_dir.')
    parser.add_argument('--split_ratio',  type=float, default=0.9,
                        help='Fraction of frames per scene for training (default: 0.9). '
                             'First 90%% = train, last 10%% = test.')
    parser.add_argument('--eta_min',      type=float, default=1e-6,
                        help='Minimum LR for CosineAnnealingLR scheduler.')
    parser.add_argument('--lambda_epi',  type=float, default=0.0,
                        help='Weight for Sampson epipolar loss on predicted matches. '
                             '0 = disabled. Try 0.1-0.5 for Phase 2.')
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
    config.LR            = args.lr
    config.TOTAL_STEPS   = args.steps
    config.LOSS_LAMBDA_C  = 1.0
    config.LOSS_LAMBDA_F  = 0.5
    config.NEG_PER_POS    = args.neg_per_pos
    config.ETA_MIN        = args.eta_min


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
        )
        val_ds = ScanNetSimpleDataset(
            args.data_dir, frame_gap=args.frame_gap, max_pairs=None,
            random_gap_range=random_gap_range, scenes=args.scenes,
            split='test', split_ratio=args.split_ratio,
        )
        print(f"Train: {len(train_ds)} | Val: {len(val_ds)} (per-scene {args.split_ratio:.0%}/{1-args.split_ratio:.0%} split)")

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                              num_workers=args.workers, collate_fn=collate_fn,
                              pin_memory=True, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=1,          shuffle=False,
                              num_workers=0, collate_fn=collate_fn)

    # ── Model ────────────────────────────────────────────────────────────────
    model = EpipolarFineTuner(config, pretrained_ckpt=args.ckpt, tau=args.tau,
                               freeze_backbone=not args.no_freeze,
                               lambda_epi=args.lambda_epi)

    # ── Callbacks & Trainer ──────────────────────────────────────────────────
    callbacks = [
        pl.callbacks.ModelCheckpoint(
            dirpath=args.checkpoint_dir,
            filename='epipolar-{step:05d}',
            every_n_train_steps=args.save_every,
            save_last=True,
            save_top_k=-1,   # keep all periodic checkpoints
        ),
        pl.callbacks.LearningRateMonitor(logging_interval='step'),
    ]
    if args.override_lr:
        callbacks.append(LROverrideCallback(args.lr))
        print(f"[Phase 2] LR override active: will set lr={args.lr} after checkpoint load")

    # Normalize precision flag: accept 'bf16' as alias for 'bf16-mixed'
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
                'tau': args.tau,
                'neg_per_pos': args.neg_per_pos,
'frame_gap': args.frame_gap,
                'random_gap': args.random_gap,
                'lambda_epi': args.lambda_epi,
                'precision': precision,
                'resume': resume_path,
            },
        )
        loggers.append(wandb_logger)
        print(f"[W&B] Logging to project='{args.wandb_project}' run='{run_name}'")

    trainer = pl.Trainer(
        max_steps=args.steps,   # --steps is always respected; use --steps 50 for quick overfit check
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=1,
        precision=precision,
        log_every_n_steps=1 if args.overfit else 10,
        # Skip validation entirely during overfitting sanity check
        limit_val_batches=0.0 if args.overfit else 1.0,
        val_check_interval=1.0,  # validate once per epoch (not every 500 steps — val is slow on Drive)
        callbacks=callbacks,
        logger=loggers if loggers else True,  # True = default CSV logger
        enable_progress_bar=True,
        gradient_clip_val=1.0,
    )

    trainer.fit(model, train_loader, val_loader, ckpt_path=resume_path)
    print(f"Training complete. Checkpoints saved to {args.checkpoint_dir}")


if __name__ == '__main__':
    main()
