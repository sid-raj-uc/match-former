"""
Benchmark on the 10% test split of a single scene.

Reproduces the exact train/val split from train_finetune.py using
torch.Generator().manual_seed(split_seed), then evaluates only on the
val (test) pairs.

Usage:
    python benchmark_test_split.py \
        --data_dir ../data/scans/scene0000_00/exported \
        --finetuned_ckpt model/weights/model-1s-1ks.ckpt
"""

import os
import sys
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config.defaultmf import get_cfg_defaults
from model.lightning_loftr import PL_LoFTR
from model.backbone.coarse_matching import CoarseMatching
from model.datasets.scannet_simple import ScanNetSimpleDataset
from model.utils.metrics import estimate_pose, relative_pose_error, error_auc
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


# ── Monkey-patch for epipolar constraint ─────────────────────────────────────
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
    mconf = input_data['mconf'].cpu().numpy() if 'mconf' in input_data else np.ones(len(mkpts0))
    return mkpts0, mkpts1, mconf


def main():
    parser = argparse.ArgumentParser(description='Benchmark on 10% test split')
    parser.add_argument('--data_dir', default='../data/scans/scene0000_00/exported')
    parser.add_argument('--vanilla_ckpt', default='model/weights/indoor-lite-LA.ckpt')
    parser.add_argument('--finetuned_ckpt', default='phase2/weights/loss-2-epi-0-2.ckpt')
    parser.add_argument('--split_seed', type=int, default=42)
    parser.add_argument('--frame_gap', type=int, default=20)
    parser.add_argument('--tau', type=float, default=10.0)
    parser.add_argument('--split_mode', default='sequential', choices=['sequential', 'random'])
    parser.add_argument('--split', default='test', choices=['train', 'test', 'all'],
                        help="'all' evaluates on the entire dataset")
    parser.add_argument('--split_ratio', type=float, default=0.9)
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
    val_ds = ScanNetSimpleDataset(args.data_dir, frame_gap=args.frame_gap, split=args.split,
                                     split_ratio=args.split_ratio,
                                     split_mode=args.split_mode, split_seed=args.split_seed)
    n_val = len(val_ds)
    print(f'\nBenchmarking on {n_val} pairs (split={args.split}, ratio={args.split_ratio}, mode={args.split_mode})\n')

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

    # ── Run benchmark ────────────────────────────────────────────────────────
    variants = {
        'Vanilla': {'model': model_vanilla, 'use_epi': False, 'thr': 0.2},
        'Vanilla+Epipolar': {'model': model_vanilla, 'use_epi': True, 'thr': 0.2},
        'Fine-Tuned': {'model': model_finetuned, 'use_epi': False, 'thr': 0.2},
        'Fine-Tuned (thr=0.05)': {'model': model_finetuned, 'use_epi': False, 'thr': 0.05},
        'Fine-Tuned (thr=0.01)': {'model': model_finetuned, 'use_epi': False, 'thr': 0.01},
    }

    results = {name: {'R_errs': [], 't_errs': [], 'n_matches': [], 'precisions': []}
               for name in variants}

    pbar = tqdm(total=len(val_ds), desc='Benchmarking', unit='pair')
    for i in range(len(val_ds)):
        item = val_ds[i]
        T0 = item['T0'].numpy()
        T1 = item['T1'].numpy()
        K = item['K'].numpy()

        if not np.isfinite(T0).all() or not np.isfinite(T1).all():
            pbar.update(1)
            continue

        T_0to1 = np.linalg.inv(T1) @ T0
        F_mat = compute_fundamental_matrix(T0, T1, K, K)

        for name, cfg in variants.items():
            model = cfg['model']
            model.matcher.coarse_matching.thr = cfg['thr']

            epi_F = F_mat if cfg['use_epi'] else None
            mkpts0, mkpts1, mconf = run_model(model, item, device, F_mat=epi_F, tau=args.tau)

            n_matches = len(mkpts0)
            ret = estimate_pose(mkpts0, mkpts1, K, K, thresh=0.5, conf=0.99999)

            if ret is None:
                results[name]['R_errs'].append(np.inf)
                results[name]['t_errs'].append(np.inf)
                results[name]['n_matches'].append(n_matches)
                results[name]['precisions'].append(0.0)
            else:
                R, t, inliers = ret
                t_err, R_err = relative_pose_error(T_0to1, R, t, ignore_gt_t_thr=0.0)
                prec = np.mean(inliers) if len(inliers) > 0 else 0.0
                results[name]['R_errs'].append(R_err)
                results[name]['t_errs'].append(t_err)
                results[name]['n_matches'].append(n_matches)
                results[name]['precisions'].append(prec)

        pbar.update(1)
    pbar.close()

    # ── Print results ────────────────────────────────────────────────────────
    print(f'\n{"="*80}')
    print(f'POSE ESTIMATION BENCHMARK — {args.data_dir} test split ({len(val_ds)} pairs)')
    print(f'{"="*80}')
    print(f'{"Method":<25} | {"AUC@5°":>7} {"AUC@10°":>8} {"AUC@20°":>8} | {"P(%)":>6} | {"Matches":>8}')
    print(f'{"-"*25}-+-{"-"*26}-+-{"-"*6}-+-{"-"*8}')

    for name in variants:
        r = results[name]
        if len(r['R_errs']) == 0:
            print(f'{name:<25} | {"N/A":>7} {"N/A":>8} {"N/A":>8} | {"N/A":>6} | {"N/A":>8}')
            continue
        pose_errors = np.maximum(np.array(r['R_errs']), np.array(r['t_errs']))
        aucs = error_auc(pose_errors, [5, 10, 20])
        print(f'{name:<25} | {aucs["auc@5"]*100:>7.2f} {aucs["auc@10"]*100:>8.2f} {aucs["auc@20"]*100:>8.2f} '
              f'| {np.mean(r["precisions"])*100:>6.1f} | {np.mean(r["n_matches"]):>8.1f}')

    print(f'{"-"*80}')
    print(f'Reference (Table 2): MatchFormer-lite-LA  |   20.42   39.23   56.82 |  87.7 |')
    print(f'{"="*80}')


if __name__ == '__main__':
    with torch.no_grad():
        main()
