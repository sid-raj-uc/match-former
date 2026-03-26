"""
Pose Estimation Benchmark (Table 2 style)
==========================================
Computes AUC @5°, @10°, @20° and matching precision P for:
  1. Vanilla  (indoor-lite-LA.ckpt)
  2. Vanilla + Epipolar constraint (τ=10)
  3. Fine-tuned (epipolar-run=50000.ckpt)

Usage:
  python pose_benchmark.py --num_pairs 100 --scenes scene0011_00 scene0012_00 scene0013_00 scene0014_00 scene0015_00
"""

import torch
import torch.nn.functional as F
import cv2
import numpy as np
import glob
import os
import argparse
from tqdm import tqdm

from config.defaultmf import get_cfg_defaults
from model.lightning_loftr import PL_LoFTR
from model.backbone.coarse_matching import CoarseMatching
from model.utils.metrics import estimate_pose, relative_pose_error, error_auc
from gt_epipolar import compute_fundamental_matrix

# Constants
H_img, W_img = 480, 640
stride = 8
H_feat, W_feat = H_img // stride, W_img // stride

# ── Epipolar mask for constrained inference ──────────────────────────────────
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

# ── Monkey-patch CoarseMatching for epipolar constraint ──────────────────────
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


def get_image_tensor(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    img = cv2.resize(img, (W_img, H_img))
    return torch.from_numpy(img).float().unsqueeze(0).unsqueeze(0) / 255.0


def get_scene_dir(scene_name, base='../data/scans'):
    """Handle scene0014 flat structure vs others with exported/."""
    exported = os.path.join(base, scene_name, 'exported')
    flat = os.path.join(base, scene_name)
    if os.path.isdir(os.path.join(exported, 'color')):
        return exported
    return flat


def run_model(model, img0, img1, device, F_mat=None, tau=10.0):
    """Run a single forward pass; return matched keypoints."""
    input_data = {'image0': img0.to(device), 'image1': img1.to(device)}
    model.matcher.coarse_matching.epipolar_F = F_mat
    if F_mat is not None:
        model.matcher.coarse_matching.epipolar_tau = tau
    with torch.no_grad():
        model.matcher(input_data)
    mkpts0 = input_data['mkpts0_f'].cpu().numpy()
    mkpts1 = input_data['mkpts1_f'].cpu().numpy()
    mconf = input_data['mconf'].cpu().numpy() if 'mconf' in input_data else np.ones(len(mkpts0))
    return mkpts0, mkpts1, mconf


def evaluate_pair(img0_path, img1_path, data_dir, K, models, device, tau=10.0):
    """
    Evaluate a single image pair across all model variants.
    Returns dict with R_err, t_err, num_matches, precision for each variant.
    """
    idx0 = os.path.basename(img0_path).split('.')[0]
    idx1 = os.path.basename(img1_path).split('.')[0]

    try:
        T0 = np.loadtxt(os.path.join(data_dir, 'pose', f'{idx0}.txt'))
        T1 = np.loadtxt(os.path.join(data_dir, 'pose', f'{idx1}.txt'))
    except FileNotFoundError:
        return None

    if not np.isfinite(T0).all() or not np.isfinite(T1).all():
        return None

    # Ground-truth relative pose (camera 0 → camera 1)
    T_0to1 = np.linalg.inv(T1) @ T0

    # Fundamental matrix for epipolar constraint
    F_mat = compute_fundamental_matrix(T0, T1, K, K)

    img0 = get_image_tensor(img0_path)
    img1 = get_image_tensor(img1_path)
    if img0 is None or img1 is None:
        return None

    results = {}

    for name, model, use_epi in models:
        epi_F = F_mat if use_epi else None
        mkpts0, mkpts1, mconf = run_model(model, img0, img1, device, F_mat=epi_F, tau=tau)

        n_matches = len(mkpts0)

        # Estimate pose via Essential matrix + RANSAC
        ret = estimate_pose(mkpts0, mkpts1, K, K, thresh=0.5, conf=0.99999)
        if ret is None:
            results[name] = {
                'R_err': np.inf, 't_err': np.inf,
                'n_matches': n_matches, 'precision': 0.0,
            }
        else:
            R, t, inliers = ret
            t_err, R_err = relative_pose_error(T_0to1, R, t, ignore_gt_t_thr=0.0)
            prec = np.mean(inliers) if len(inliers) > 0 else 0.0
            results[name] = {
                'R_err': R_err, 't_err': t_err,
                'n_matches': n_matches, 'precision': prec,
            }

    return results


def compute_auc_and_precision(all_results, model_name):
    """Compute AUC @5/10/20 and mean precision for a given model variant."""
    R_errs = [r[model_name]['R_err'] for r in all_results]
    t_errs = [r[model_name]['t_err'] for r in all_results]
    precs = [r[model_name]['precision'] for r in all_results]
    n_matches = [r[model_name]['n_matches'] for r in all_results]

    # Pose error = max(R_err, t_err), same as LoFTR / MatchFormer eval
    pose_errors = np.maximum(np.array(R_errs), np.array(t_errs))
    aucs = error_auc(pose_errors, [5, 10, 20])

    return {
        'auc@5': aucs['auc@5'] * 100,
        'auc@10': aucs['auc@10'] * 100,
        'auc@20': aucs['auc@20'] * 100,
        'precision': np.mean(precs) * 100,
        'avg_matches': np.mean(n_matches),
        'n_pairs': len(all_results),
    }


def main():
    parser = argparse.ArgumentParser(description='Pose Estimation Benchmark (Table 2 style)')
    parser.add_argument('--num_pairs', type=int, default=100,
                        help='Number of pairs per scene')
    parser.add_argument('--pair_gap', type=int, default=20,
                        help='Frame gap between image pairs')
    parser.add_argument('--scenes', nargs='+',
                        default=['scene0011_00', 'scene0012_00', 'scene0013_00',
                                 'scene0014_00', 'scene0015_00'])
    parser.add_argument('--data_dir', type=str, default='../data/scans',
                        help='Root directory containing scene subdirs')
    parser.add_argument('--tau', type=float, default=10.0,
                        help='Epipolar mask softness for vanilla+epipolar')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--vanilla_ckpt', type=str, default='model/weights/indoor-lite-LA.ckpt')
    parser.add_argument('--finetuned_ckpt', type=str, default='model/weights/epipolar-run=50000.ckpt')
    parser.add_argument('--sweep_thresholds', nargs='+', type=float,
                        default=[0.05, 0.02, 0.01, 0.005],
                        help='Thresholds to sweep for fine-tuned model')
    args = parser.parse_args()

    np.random.seed(args.seed)

    # Device
    if torch.cuda.is_available():
        device = 'cuda'
    elif torch.backends.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'
    print(f'Device: {device}')

    # Model config
    config = get_cfg_defaults()
    config.MATCHFORMER.BACKBONE_TYPE = 'litela'
    config.MATCHFORMER.SCENS = 'indoor'
    config.MATCHFORMER.RESOLUTION = (8, 4)
    config.MATCHFORMER.COARSE.D_MODEL = 192
    config.MATCHFORMER.COARSE.D_FFN = 192

    # Load models
    print('Loading models...')
    model_vanilla = PL_LoFTR(config, pretrained_ckpt=args.vanilla_ckpt).to(device).eval()
    model_constrained = PL_LoFTR(config, pretrained_ckpt=args.vanilla_ckpt).to(device).eval()
    model_finetuned = PL_LoFTR(config, pretrained_ckpt=args.finetuned_ckpt).to(device).eval()

    sweep_thrs = sorted(args.sweep_thresholds)

    # ── Collect image pairs first (scene-aware) ─────────────────────────────
    # We collect pairs once, then run each threshold on the same pairs
    all_pair_info = []  # list of (img0_path, img1_path, data_dir, K, scene_name)

    for scene in args.scenes:
        data_dir = get_scene_dir(scene, base=args.data_dir)
        all_imgs = sorted(glob.glob(f'{data_dir}/color/*.jpg'),
                          key=lambda x: int(os.path.basename(x).split('.')[0]))
        K_intr = np.loadtxt(f'{data_dir}/intrinsic/intrinsic_depth.txt')[:3, :3]
        n_imgs = len(all_imgs)
        print(f'Scene: {scene}  ({n_imgs} frames)')

        collected = 0
        attempts = 0
        while collected < args.num_pairs and attempts < args.num_pairs * 5:
            attempts += 1
            idx1 = np.random.randint(0, max(1, n_imgs - args.pair_gap - 1))
            idx2 = idx1 + args.pair_gap
            if idx2 >= n_imgs:
                continue
            # Quick validity check — poses exist and are finite
            idx0_num = os.path.basename(all_imgs[idx1]).split('.')[0]
            idx1_num = os.path.basename(all_imgs[idx2]).split('.')[0]
            p0 = os.path.join(data_dir, 'pose', f'{idx0_num}.txt')
            p1 = os.path.join(data_dir, 'pose', f'{idx1_num}.txt')
            if not os.path.exists(p0) or not os.path.exists(p1):
                continue
            T0 = np.loadtxt(p0); T1 = np.loadtxt(p1)
            if not np.isfinite(T0).all() or not np.isfinite(T1).all():
                continue
            all_pair_info.append((all_imgs[idx1], all_imgs[idx2], data_dir, K_intr, scene))
            collected += 1

    print(f'\nCollected {len(all_pair_info)} valid pairs total.\n')

    # ── Run vanilla & vanilla+epipolar at default thr=0.2 ───────────────────
    print('='*75)
    print('Running Vanilla & Vanilla+Epipolar (thr=0.2) ...')
    print('='*75)

    vanilla_results = []
    model_vanilla.matcher.coarse_matching.thr = 0.2
    model_constrained.matcher.coarse_matching.thr = 0.2

    base_variants = [
        ('Vanilla', model_vanilla, False),
        ('Vanilla+Epipolar', model_constrained, True),
    ]

    pbar = tqdm(total=len(all_pair_info), desc='Vanilla/Epi', unit='pair')
    for img0_path, img1_path, data_dir, K_intr, scene in all_pair_info:
        res = evaluate_pair(img0_path, img1_path, data_dir, K_intr,
                            base_variants, device, tau=args.tau)
        if res is not None:
            vanilla_results.append(res)
        pbar.update(1)
    pbar.close()

    print(f'\n{"Method":<25} | {"AUC@5°":>7} {"AUC@10°":>8} {"AUC@20°":>8} | {"P(%)":>6} | {"Matches":>8}')
    print(f'{"-"*25}-+-{"-"*26}-+-{"-"*6}-+-{"-"*8}')
    for name, _, _ in base_variants:
        m = compute_auc_and_precision(vanilla_results, name)
        print(f'{name:<25} | {m["auc@5"]:>7.2f} {m["auc@10"]:>8.2f} {m["auc@20"]:>8.2f} '
              f'| {m["precision"]:>6.1f} | {m["avg_matches"]:>8.1f}')

    # ── Sweep thresholds for fine-tuned model ────────────────────────────────
    print(f'\n{"="*75}')
    print(f'THRESHOLD SWEEP — Fine-Tuned (run=50000)')
    print(f'Thresholds: {sweep_thrs}')
    print(f'{"="*75}')

    ft_variant = [('Fine-Tuned', model_finetuned, False)]

    for thr in sweep_thrs:
        model_finetuned.matcher.coarse_matching.thr = thr
        ft_results = []

        pbar = tqdm(total=len(all_pair_info), desc=f'thr={thr}', unit='pair')
        for img0_path, img1_path, data_dir, K_intr, scene in all_pair_info:
            res = evaluate_pair(img0_path, img1_path, data_dir, K_intr,
                                ft_variant, device, tau=args.tau)
            if res is not None:
                ft_results.append(res)
            pbar.update(1)
        pbar.close()

        m = compute_auc_and_precision(ft_results, 'Fine-Tuned')
        print(f'  thr={thr:<6} | AUC@5°: {m["auc@5"]:>6.2f}  AUC@10°: {m["auc@10"]:>6.2f}  '
              f'AUC@20°: {m["auc@20"]:>6.2f}  |  P: {m["precision"]:>5.1f}%  |  Matches: {m["avg_matches"]:>7.1f}')

    # ── Reference ────────────────────────────────────────────────────────────
    print(f'\n{"─"*75}')
    print(f'Reference (Table 2 - ScanNet test set):')
    print(f'{"MatchFormer-lite-LA":<25} | {"20.42":>7} {"39.23":>8} {"56.82":>8} | {"87.7":>6} |')
    print(f'{"─"*75}')


if __name__ == '__main__':
    with torch.no_grad():
        main()
