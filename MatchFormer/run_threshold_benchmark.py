"""
Threshold benchmark: compares vanilla model vs fine-tuned model at several
confidence thresholds in a single pass over 100 ScanNet pairs.

Usage:
    python run_threshold_benchmark.py
    python run_threshold_benchmark.py --num_pairs 50
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

H_img, W_img = 480, 640

# ── monkey-patch (no epipolar; we only want threshold control here) ──────────
def plain_forward(self, feat_c0, feat_c1, data, mask_c0=None, mask_c1=None):
    feat_c0_n = feat_c0 / feat_c0.shape[-1] ** .5
    feat_c1_n = feat_c1 / feat_c1.shape[-1] ** .5
    sim = torch.einsum('nlc,nsc->nls', feat_c0_n, feat_c1_n) / self.temperature
    if mask_c0 is not None:
        sim.masked_fill_(~(mask_c0[..., None] * mask_c1[:, None]).bool(), -1e9)
    conf = F.softmax(sim, 1) * F.softmax(sim, 2)
    data.update({'conf_matrix': conf})
    data.update(**self.get_coarse_match(conf, data))

CoarseMatching.forward = plain_forward

# ── helpers (identical to run_benchmark.py) ──────────────────────────────────
def get_image_tensor(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None: return None
    img = cv2.resize(img, (W_img, H_img))
    return torch.from_numpy(img).float().unsqueeze(0).unsqueeze(0) / 255.0

def get_gt_matches(mkpts0, depth_path, T1, T2, K):
    depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
    if depth is None:
        return np.zeros(len(mkpts0), dtype=bool), np.zeros_like(mkpts0)
    depth = depth.astype(float) / 1000.0
    fx, fy, cx, cy = K[0,0], K[1,1], K[0,2], K[1,2]
    T_cv2gl = np.array([[1,0,0,0],[0,-1,0,0],[0,0,-1,0],[0,0,0,1]])
    T_12 = T_cv2gl @ np.linalg.inv(T2) @ T1 @ T_cv2gl
    valid = np.zeros(len(mkpts0), dtype=bool)
    gt_pts = np.zeros_like(mkpts0)
    for i, pt in enumerate(mkpts0):
        xi, yi = int(round(pt[0])), int(round(pt[1]))
        if not (0 <= yi < depth.shape[0] and 0 <= xi < depth.shape[1]): continue
        z = depth[yi, xi]
        if z <= 0.1 or z > 10.0: continue
        p = T_12 @ np.array([(pt[0]-cx)*z/fx, (pt[1]-cy)*z/fy, z, 1.0])
        if p[2] <= 0: continue
        u, v = (p[0]*fx/p[2])+cx, (p[1]*fy/p[2])+cy
        if 0 <= u < W_img and 0 <= v < H_img:
            valid[i], gt_pts[i] = True, [u, v]
    return valid, gt_pts

def compute_metrics(mkpts0, mkpts1, depth_path, T1, T2, K):
    if len(mkpts0) == 0:
        return {'total': 0, 'mean_err': 0.0, 'p3': 0.0, 'p5': 0.0}
    valid, gt = get_gt_matches(mkpts0, depth_path, T1, T2, K)
    mk0v, mk1v, gtv = mkpts0[valid], mkpts1[valid], gt[valid]
    if len(mk0v) == 0:
        return {'total': 0, 'mean_err': 0.0, 'p3': 0.0, 'p5': 0.0}
    errs = np.linalg.norm(mk1v - gtv, axis=1)
    return {
        'total':    len(errs),
        'mean_err': float(np.mean(errs)),
        'p3':       float(np.mean(errs < 3.0)),
        'p5':       float(np.mean(errs < 5.0)),
    }

def run_inference(model, img0_t, img1_t, thr):
    model.matcher.coarse_matching.thr = thr
    data = {'image0': img0_t, 'image1': img1_t}
    with torch.no_grad():
        model.matcher(data)
    return data['mkpts0_f'].cpu().numpy(), data['mkpts1_f'].cpu().numpy()

# ── main ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_pairs', type=int, default=100)
    parser.add_argument('--vanilla_ckpt', type=str, default='model/weights/indoor-lite-LA.ckpt')
    parser.add_argument('--ft_ckpt',      type=str, default='model/weights/last.ckpt')
    args = parser.parse_args()

    config = get_cfg_defaults()
    config.MATCHFORMER.BACKBONE_TYPE = 'litela'
    config.MATCHFORMER.SCENS         = 'indoor'
    config.MATCHFORMER.RESOLUTION    = (8, 4)
    config.MATCHFORMER.COARSE.D_MODEL = 192
    config.MATCHFORMER.COARSE.D_FFN   = 192

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_v  = PL_LoFTR(config, pretrained_ckpt=args.vanilla_ckpt).to(device).eval()
    model_ft = PL_LoFTR(config, pretrained_ckpt=args.ft_ckpt).to(device).eval()

    data_dir = '../data/scans/scene0000_00/exported'
    all_imgs = sorted(glob.glob(os.path.join(data_dir, 'color', '*.jpg')),
                      key=lambda x: int(os.path.basename(x).split('.')[0]))
    K = np.loadtxt(os.path.join(data_dir, 'intrinsic', 'intrinsic_depth.txt'))[:3, :3]

    VANILLA_THR = 0.2
    ft_thresholds = [0.2, 0.1, 0.05, 0.02, 0.01]
    pair_gap = 20

    # Accumulators: vanilla + one per ft threshold
    vanilla_results = []
    ft_results = {thr: [] for thr in ft_thresholds}

    print(f'\n--- Threshold Benchmark: {args.num_pairs} pairs ---')
    print(f'Vanilla checkpoint : {args.vanilla_ckpt}  (thr={VANILLA_THR})')
    print(f'Fine-tuned ckpt    : {args.ft_ckpt}')
    print(f'Fine-tuned thr sweep: {ft_thresholds}\n')

    collected, idx = 0, 0
    pbar = tqdm(total=args.num_pairs)

    while collected < args.num_pairs and idx < len(all_imgs) - pair_gap:
        path0 = all_imgs[idx]
        path1 = all_imgs[idx + pair_gap]
        idx += 1

        num0 = os.path.basename(path0).split('.')[0]
        num1 = os.path.basename(path1).split('.')[0]

        try:
            T0 = np.loadtxt(os.path.join(data_dir, 'pose', f'{num0}.txt'))
            T1 = np.loadtxt(os.path.join(data_dir, 'pose', f'{num1}.txt'))
        except FileNotFoundError:
            continue
        if not (np.isfinite(T0).all() and np.isfinite(T1).all()):
            continue

        img0_t = get_image_tensor(path0)
        img1_t = get_image_tensor(path1)
        if img0_t is None or img1_t is None:
            continue

        depth_path = os.path.join(data_dir, 'depth', f'{num0}.png')

        # ── vanilla ──────────────────────────────────────────────────────────
        mk0_v, mk1_v = run_inference(model_v, img0_t, img1_t, VANILLA_THR)
        m_v = compute_metrics(mk0_v, mk1_v, depth_path, T0, T1, K)
        if m_v['total'] == 0:
            continue   # skip pairs with no valid depth-verified matches

        vanilla_results.append(m_v)

        # ── fine-tuned at each threshold ──────────────────────────────────────
        for thr in ft_thresholds:
            mk0_ft, mk1_ft = run_inference(model_ft, img0_t, img1_t, thr)
            ft_results[thr].append(compute_metrics(mk0_ft, mk1_ft, depth_path, T0, T1, K))

        collected += 1
        pbar.update(1)

    pbar.close()

    # ── aggregate ─────────────────────────────────────────────────────────────
    def agg(rows):
        if not rows: return {'total': 0, 'mean_err': 0, 'p3': 0, 'p5': 0}
        return {
            'total':    np.mean([r['total']    for r in rows]),
            'mean_err': np.mean([r['mean_err'] for r in rows]),
            'p3':       np.mean([r['p3']       for r in rows]),
            'p5':       np.mean([r['p5']       for r in rows]),
        }

    v = agg(vanilla_results)

    print('\n' + '='*70)
    print('BENCHMARK RESULTS')
    print('='*70)
    print(f'\n{"Model":<35} {"Mean Err":>10} {"P@3px":>8} {"P@5px":>8} {"Avg Matches":>12}')
    print('-'*70)
    print(f'{"Vanilla (thr=0.20)":<35} {v["mean_err"]:>10.2f} {v["p3"]:>7.2%} {v["p5"]:>7.2%} {v["total"]:>12.1f}')

    for thr in ft_thresholds:
        ft = agg(ft_results[thr])
        label = f'Fine-Tuned (thr={thr})'
        # compute improvement vs vanilla
        err_delta = (v["mean_err"] - ft["mean_err"]) / v["mean_err"] * 100
        print(f'{label:<35} {ft["mean_err"]:>10.2f} {ft["p3"]:>7.2%} {ft["p5"]:>7.2%} {ft["total"]:>12.1f}   err↓{err_delta:.0f}%')

    print('='*70)


if __name__ == '__main__':
    with torch.no_grad():
        main()
