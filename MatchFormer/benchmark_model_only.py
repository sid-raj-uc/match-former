"""
Benchmark a single model on the 10% test split.
Only runs the specified checkpoint — no baselines.

Usage:
    python benchmark_model_only.py \
        --data_dir ../data/scans/scene0000_00/exported \
        --ckpt model/weights/my_model.ckpt
"""

import os
import sys
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import random_split
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config.defaultmf import get_cfg_defaults
from model.lightning_loftr import PL_LoFTR
from model.backbone.coarse_matching import CoarseMatching
from model.datasets.scannet_simple import ScanNetSimpleDataset
from model.utils.metrics import estimate_pose, relative_pose_error, error_auc
from gt_epipolar import compute_fundamental_matrix

H_img, W_img = 480, 640


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


def run_model(model, item, device, F_mat=None, tau=10.0):
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


def get_gt_reprojection(mkpts0, depth0, T0, T1, K):
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    H, W = depth0.shape
    x_idx = np.round(mkpts0[:, 0]).astype(int)
    y_idx = np.round(mkpts0[:, 1]).astype(int)
    valid = (x_idx >= 0) & (x_idx < W) & (y_idx >= 0) & (y_idx < H)
    z = np.zeros(len(mkpts0), dtype=np.float32)
    z[valid] = depth0[y_idx[valid], x_idx[valid]]
    valid &= (z > 0.1) & (z <= 10.0)
    x_c = (mkpts0[:, 0] - cx) * z / fx
    y_c = (mkpts0[:, 1] - cy) * z / fy
    pts_h = np.stack([x_c, y_c, z, np.ones(len(mkpts0))], axis=1)
    T_0to1 = np.linalg.inv(T1) @ T0
    pts_c1 = (T_0to1 @ pts_h.T).T
    valid &= pts_c1[:, 2] > 0
    u1 = np.where(valid, pts_c1[:, 0] * fx / np.where(valid, pts_c1[:, 2], 1) + cx, 0)
    v1 = np.where(valid, pts_c1[:, 1] * fy / np.where(valid, pts_c1[:, 2], 1) + cy, 0)
    valid &= (u1 >= 0) & (u1 < W_img) & (v1 >= 0) & (v1 < H_img)
    return valid, np.stack([u1, v1], axis=1)


def main():
    parser = argparse.ArgumentParser(description='Benchmark a model on 10% test split')
    parser.add_argument('--data_dir', default='../data/scans/scene0000_00/exported')
    parser.add_argument('--ckpt', required=True, help='Checkpoint to evaluate')
    parser.add_argument('--split_seed', type=int, default=42)
    parser.add_argument('--frame_gap', type=int, default=20)
    parser.add_argument('--tau', type=float, default=10.0)
    parser.add_argument('--thresholds', nargs='+', type=float, default=[0.2, 0.05, 0.01])
    args = parser.parse_args()

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
    _, val_ds = random_split(dataset, [n_train, n_val], generator=split_generator)
    print(f'Split: {n_train} train / {n_val} val (seed={args.split_seed})')

    # ── Model ────────────────────────────────────────────────────────────────
    config = get_cfg_defaults()
    config.MATCHFORMER.BACKBONE_TYPE = 'litela'
    config.MATCHFORMER.SCENS = 'indoor'
    config.MATCHFORMER.RESOLUTION = (8, 4)
    config.MATCHFORMER.COARSE.D_MODEL = 192
    config.MATCHFORMER.COARSE.D_FFN = 192

    ckpt_name = os.path.basename(args.ckpt)
    print(f'Model: {ckpt_name}\n')
    model = PL_LoFTR(config, pretrained_ckpt=args.ckpt).to(device).eval()

    # ── Evaluate at each threshold ───────────────────────────────────────────
    for thr in sorted(args.thresholds, reverse=True):
        model.matcher.coarse_matching.thr = thr

        pose_results = {'R_errs': [], 't_errs': [], 'n_matches': [], 'precisions': []}
        dist_errors = []
        dist_n_matches = []
        dist_n_valid = []

        pbar = tqdm(total=len(val_ds), desc=f'thr={thr}', unit='pair')
        for i in range(len(val_ds)):
            item = val_ds[i]
            T0 = item['T0'].numpy()
            T1 = item['T1'].numpy()
            K = item['K'].numpy()
            depth0 = item['depth0'].numpy()

            if not np.isfinite(T0).all() or not np.isfinite(T1).all():
                pbar.update(1)
                continue

            mkpts0, mkpts1 = run_model(model, item, device)

            # ── Pose AUC ─────────────────────────────────────────────────
            T_0to1 = np.linalg.inv(T1) @ T0
            n_matches = len(mkpts0)
            ret = estimate_pose(mkpts0, mkpts1, K, K, thresh=0.5, conf=0.99999)
            if ret is None:
                pose_results['R_errs'].append(np.inf)
                pose_results['t_errs'].append(np.inf)
                pose_results['n_matches'].append(n_matches)
                pose_results['precisions'].append(0.0)
            else:
                R, t, inliers = ret
                t_err, R_err = relative_pose_error(T_0to1, R, t, ignore_gt_t_thr=0.0)
                prec = np.mean(inliers) if len(inliers) > 0 else 0.0
                pose_results['R_errs'].append(R_err)
                pose_results['t_errs'].append(t_err)
                pose_results['n_matches'].append(n_matches)
                pose_results['precisions'].append(prec)

            # ── Reprojection distance ────────────────────────────────────
            if len(mkpts0) > 0:
                valid, gt_mkpts1 = get_gt_reprojection(mkpts0, depth0, T0, T1, K)
                errs = np.linalg.norm(mkpts1[valid] - gt_mkpts1[valid], axis=1)
                dist_errors.append(errs)
                dist_n_matches.append(n_matches)
                dist_n_valid.append(int(valid.sum()))
            else:
                dist_errors.append(np.array([]))
                dist_n_matches.append(0)
                dist_n_valid.append(0)

            pbar.update(1)
        pbar.close()

        # ── Print ────────────────────────────────────────────────────────
        R_errs = np.array(pose_results['R_errs'])
        t_errs = np.array(pose_results['t_errs'])
        pose_errors = np.maximum(R_errs, t_errs)
        aucs = error_auc(pose_errors, [5, 10, 20])

        all_dist = np.concatenate(dist_errors) if any(len(e) > 0 for e in dist_errors) else np.array([])

        print(f'\n  thr={thr}  |  {ckpt_name}')
        print(f'  Pose AUC   @5°: {aucs["auc@5"]*100:6.2f}  @10°: {aucs["auc@10"]*100:6.2f}  @20°: {aucs["auc@20"]*100:6.2f}  |  P: {np.mean(pose_results["precisions"])*100:5.1f}%  |  Matches: {np.mean(pose_results["n_matches"]):.0f}')
        if len(all_dist) > 0:
            print(f'  Distance   Mean: {np.mean(all_dist):.2f}px  Med: {np.median(all_dist):.2f}px  |  P@3px: {np.mean(all_dist<3)*100:5.1f}%  P@5px: {np.mean(all_dist<5)*100:5.1f}%  |  Valid: {sum(dist_n_valid)}')
        print()

    # ── Reference baselines (from previous run) ─────────────────────────────
    print(f'{"─"*80}')
    print(f'Reference baselines (scene0000, 555 val pairs):')
    print(f'  Vanilla           Pose AUC @5°: 23.03  @10°: 43.65  @20°: 64.54  |  P: 64.8%  |  Matches: 2879')
    print(f'                    Distance Mean: 2.21px  Med: 1.77px  |  P@3px: 77.9%  P@5px: 93.9%')
    print(f'  Vanilla+Epipolar  Pose AUC @5°: 22.81  @10°: 43.97  @20°: 65.83  |  P: 66.1%  |  Matches: 2604')
    print(f'                    Distance Mean: 2.11px  Med: 1.72px  |  P@3px: 79.9%  P@5px: 95.1%')
    print(f'{"─"*80}')


if __name__ == '__main__':
    with torch.no_grad():
        main()
