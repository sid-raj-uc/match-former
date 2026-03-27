"""
Reprojection Distance Benchmark on the 10% test split.

Measures how far predicted matches in image1 are from their GT reprojected
locations (mkpts0 → unproject with depth0 → transform to cam1 → reproject).

Reproduces the exact train/val split from train_finetune.py using seed=42.
"""

import os
import sys
import argparse
import numpy as np
import cv2
import torch
import torch.nn.functional as F
from torch.utils.data import random_split
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config.defaultmf import get_cfg_defaults
from model.lightning_loftr import PL_LoFTR
from model.backbone.coarse_matching import CoarseMatching
from model.datasets.scannet_simple import ScanNetSimpleDataset
from gt_epipolar import compute_fundamental_matrix

H_img, W_img = 480, 640


# ── Epipolar mask ────────────────────────────────────────────────────────────
def get_epipolar_mask_matrix(F_mat, H0, W0, H1, W1, tau=10.0, device='cpu'):
    y0, x0 = torch.meshgrid(torch.arange(H0), torch.arange(W0), indexing='ij')
    x0_img = (x0.float() / W0) * W_img
    y0_img = (y0.float() / H0) * H_img
    y1, x1 = torch.meshgrid(torch.arange(H1), torch.arange(W1), indexing='ij')
    x1_img = (x1.float() / W1) * W_img
    y1_img = (y1.float() / H1) * H_img
    p0 = torch.stack([x0_img.flatten(), y0_img.flatten(), torch.ones_like(x0_img.flatten())], dim=1).to(device)
    p1 = torch.stack([x1_img.flatten(), y1_img.flatten(), torch.ones_like(x1_img.flatten())], dim=1).to(device)
    F_t = torch.tensor(F_mat, dtype=torch.float32, device=device)
    l_prime = p0 @ F_t.T
    num = torch.abs(l_prime @ p1.T)
    denom = torch.sqrt(l_prime[:, 0]**2 + l_prime[:, 1]**2).unsqueeze(1)
    distances = num / (denom + 1e-8)
    mask = torch.exp(-distances / tau)
    return mask.unsqueeze(0)


# ── Monkey-patch CoarseMatching ──────────────────────────────────────────────
original_forward = CoarseMatching.forward

def constrained_forward(self, feat_c0, feat_c1, data, mask_c0=None, mask_c1=None):
    N, L, S, C = feat_c0.size(0), feat_c0.size(1), feat_c1.size(1), feat_c0.size(2)
    feat_c0_norm, feat_c1_norm = map(lambda feat: feat / (feat.shape[-1]**.5), [feat_c0, feat_c1])
    if self.match_type == 'dual_softmax':
        sim_matrix = torch.einsum("nlc,nsc->nls", feat_c0_norm, feat_c1_norm) / self.temperature
        if mask_c0 is not None:
            sim_matrix.masked_fill_(~(mask_c0[..., None] * mask_c1[:, None]).bool(), -1e9)
        conf_matrix = F.softmax(sim_matrix, 1) * F.softmax(sim_matrix, 2)
        if getattr(self, 'epipolar_F', None) is not None:
            tau = getattr(self, 'epipolar_tau', 10.0)
            H0, W0 = data['hw0_c']
            H1, W1 = data['hw1_c']
            epi_mask = get_epipolar_mask_matrix(
                self.epipolar_F, H0, W0, H1, W1, tau=tau, device=conf_matrix.device)
            conf_matrix = conf_matrix * epi_mask
    data.update({'conf_matrix': conf_matrix})
    data.update(**self.get_coarse_match(conf_matrix, data))

CoarseMatching.forward = constrained_forward


def get_gt_reprojection(mkpts0, depth0, T0, T1, K):
    """
    Reproject mkpts0 from image0 → 3D (using depth0) → image1.

    Args:
        mkpts0: [N, 2] predicted keypoints in image 0 (x, y pixel coords)
        depth0: [H, W] depth map for image 0 in metres
        T0: [4, 4] cam0-to-world pose
        T1: [4, 4] cam1-to-world pose
        K:  [3, 3] intrinsic matrix

    Returns:
        valid: [N] bool mask — which points have valid depth + reprojection
        gt_mkpts1: [N, 2] GT pixel coords in image 1
    """
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    H, W = depth0.shape

    # Sample depth at predicted keypoint locations
    x_idx = np.round(mkpts0[:, 0]).astype(int)
    y_idx = np.round(mkpts0[:, 1]).astype(int)

    valid = (x_idx >= 0) & (x_idx < W) & (y_idx >= 0) & (y_idx < H)
    z = np.zeros(len(mkpts0), dtype=np.float32)
    z[valid] = depth0[y_idx[valid], x_idx[valid]]
    valid &= (z > 0.1) & (z <= 10.0)

    # Unproject to 3D in camera 0
    x_c = (mkpts0[:, 0] - cx) * z / fx
    y_c = (mkpts0[:, 1] - cy) * z / fy
    pts_h = np.stack([x_c, y_c, z, np.ones(len(mkpts0))], axis=1)  # [N, 4]

    # Transform: camera 0 → world → camera 1
    T_0to1 = np.linalg.inv(T1) @ T0  # [4, 4]
    pts_c1 = (T_0to1 @ pts_h.T).T  # [N, 4]

    # Must be in front of camera 1
    valid &= pts_c1[:, 2] > 0

    # Project to image 1
    u1 = np.where(valid, pts_c1[:, 0] * fx / np.where(valid, pts_c1[:, 2], 1) + cx, 0)
    v1 = np.where(valid, pts_c1[:, 1] * fy / np.where(valid, pts_c1[:, 2], 1) + cy, 0)

    # Must land within image bounds
    valid &= (u1 >= 0) & (u1 < W_img) & (v1 >= 0) & (v1 < H_img)

    gt_mkpts1 = np.stack([u1, v1], axis=1)
    return valid, gt_mkpts1


def run_model(model, item, device, F_mat=None, tau=10.0):
    """Run inference on a single dataset item."""
    input_data = {
        'image0': item['image0'].unsqueeze(0).to(device),
        'image1': item['image1'].unsqueeze(0).to(device),
    }
    model.matcher.coarse_matching.epipolar_F = F_mat
    if F_mat is not None:
        model.matcher.coarse_matching.epipolar_tau = tau
    with torch.no_grad():
        model.matcher(input_data)
    mkpts0 = input_data['mkpts0_f'].cpu().numpy()
    mkpts1 = input_data['mkpts1_f'].cpu().numpy()
    return mkpts0, mkpts1


def main():
    parser = argparse.ArgumentParser(description='Reprojection distance benchmark on test split')
    parser.add_argument('--data_dir', default='../data/scans/scene0000_00/exported')
    parser.add_argument('--vanilla_ckpt', default='model/weights/indoor-lite-LA.ckpt')
    parser.add_argument('--finetuned_ckpt', default='model/weights/model-1s-1ks.ckpt')
    parser.add_argument('--split_seed', type=int, default=42)
    parser.add_argument('--frame_gap', type=int, default=20)
    parser.add_argument('--tau', type=float, default=10.0)
    args = parser.parse_args()

    # Device
    if torch.cuda.is_available():
        device = 'cuda'
    elif torch.backends.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'
    print(f'Device: {device}')

    # ── Reproduce train/val split ────────────────────────────────────────────
    dataset = ScanNetSimpleDataset(args.data_dir, frame_gap=args.frame_gap)
    n_val = max(1, int(len(dataset) * 0.1))
    n_train = len(dataset) - n_val
    split_generator = torch.Generator().manual_seed(args.split_seed)
    train_ds, val_ds = random_split(dataset, [n_train, n_val], generator=split_generator)
    print(f'\nSplit: {n_train} train / {n_val} val (seed={args.split_seed})')
    print(f'Benchmarking on {n_val} val pairs\n')

    # ── Model config ─────────────────────────────────────────────────────────
    config = get_cfg_defaults()
    config.MATCHFORMER.BACKBONE_TYPE = 'litela'
    config.MATCHFORMER.SCENS = 'indoor'
    config.MATCHFORMER.RESOLUTION = (8, 4)
    config.MATCHFORMER.COARSE.D_MODEL = 192
    config.MATCHFORMER.COARSE.D_FFN = 192

    # ── Load models ──────────────────────────────────────────────────────────
    print('Loading models...')
    model_vanilla = PL_LoFTR(config, pretrained_ckpt=args.vanilla_ckpt).to(device).eval()
    model_finetuned = PL_LoFTR(config, pretrained_ckpt=args.finetuned_ckpt).to(device).eval()

    # ── Define variants ──────────────────────────────────────────────────────
    variants = {
        'Vanilla': {'model': model_vanilla, 'use_epi': False, 'thr': 0.2},
        'Vanilla+Epipolar': {'model': model_vanilla, 'use_epi': True, 'thr': 0.2},
        'Fine-Tuned (thr=0.2)': {'model': model_finetuned, 'use_epi': False, 'thr': 0.2},
        'Fine-Tuned (thr=0.05)': {'model': model_finetuned, 'use_epi': False, 'thr': 0.05},
        'Fine-Tuned (thr=0.01)': {'model': model_finetuned, 'use_epi': False, 'thr': 0.01},
    }

    # Per-variant accumulators
    stats = {name: {'errors': [], 'n_matches': [], 'n_valid': []} for name in variants}

    pbar = tqdm(total=len(val_ds), desc='Benchmarking', unit='pair')
    for i in range(len(val_ds)):
        item = val_ds[i]
        T0 = item['T0'].numpy()
        T1 = item['T1'].numpy()
        K = item['K'].numpy()
        depth0 = item['depth0'].numpy()  # [H, W] in metres

        if not np.isfinite(T0).all() or not np.isfinite(T1).all():
            pbar.update(1)
            continue

        F_mat = compute_fundamental_matrix(T0, T1, K, K)

        for name, cfg in variants.items():
            model = cfg['model']
            model.matcher.coarse_matching.thr = cfg['thr']

            epi_F = F_mat if cfg['use_epi'] else None
            mkpts0, mkpts1 = run_model(model, item, device, F_mat=epi_F, tau=args.tau)

            if len(mkpts0) == 0:
                stats[name]['errors'].append(np.array([]))
                stats[name]['n_matches'].append(0)
                stats[name]['n_valid'].append(0)
                continue

            valid, gt_mkpts1 = get_gt_reprojection(mkpts0, depth0, T0, T1, K)

            mkpts1_valid = mkpts1[valid]
            gt_valid = gt_mkpts1[valid]
            errs = np.linalg.norm(mkpts1_valid - gt_valid, axis=1)

            stats[name]['errors'].append(errs)
            stats[name]['n_matches'].append(len(mkpts0))
            stats[name]['n_valid'].append(int(valid.sum()))

        pbar.update(1)
    pbar.close()

    # ── Print results ────────────────────────────────────────────────────────
    print(f'\n{"="*85}')
    print(f'REPROJECTION DISTANCE BENCHMARK — scene0000 test split ({len(val_ds)} pairs)')
    print(f'{"="*85}')
    print(f'{"Method":<25} | {"Mean Err":>9} | {"Med Err":>8} | {"P@3px":>7} | {"P@5px":>7} | {"Matches":>8} | {"Valid":>6}')
    print(f'{"-"*25}-+-{"-"*9}-+-{"-"*8}-+-{"-"*7}-+-{"-"*7}-+-{"-"*8}-+-{"-"*6}')

    for name in variants:
        s = stats[name]
        all_errs = np.concatenate(s['errors']) if any(len(e) > 0 for e in s['errors']) else np.array([])
        total_matches = sum(s['n_matches'])
        total_valid = sum(s['n_valid'])

        if len(all_errs) == 0:
            print(f'{name:<25} | {"N/A":>9} | {"N/A":>8} | {"N/A":>7} | {"N/A":>7} | {total_matches:>8} | {total_valid:>6}')
            continue

        mean_err = np.mean(all_errs)
        med_err = np.median(all_errs)
        p3 = np.mean(all_errs < 3.0) * 100
        p5 = np.mean(all_errs < 5.0) * 100

        print(f'{name:<25} | {mean_err:>8.2f}px | {med_err:>7.2f}px | {p3:>6.1f}% | {p5:>6.1f}% | {total_matches:>8} | {total_valid:>6}')

    print(f'{"="*85}')


if __name__ == '__main__':
    with torch.no_grad():
        main()
