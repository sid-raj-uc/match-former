"""
WxBS Pose Estimation Benchmark — Vanilla MatchFormer.

Since WxBS has no camera intrinsics or GT poses, we:
  1. Assume default K: f = max(H, W), cx = W/2, cy = H/2  (applied after resize to 640x480)
  2. Derive pseudo-GT pose (R_gt, t_gt) from GT correspondences via findEssentialMat + recoverPose
  3. Estimate pose from predicted matches the same way
  4. Measure angular errors in R and t, report AUC @ 5° / 10° / 20°

Usage:
    python benchmark_wxbs_pose.py --data_dir /path/to/WxBS_data_folder/v1.1
"""

import os
import sys
import argparse
import numpy as np
import cv2
import torch
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config.defaultmf import get_cfg_defaults
from model.lightning_loftr import PL_LoFTR
from model.utils.metrics import estimate_pose, error_auc

TARGET_H, TARGET_W = 480, 640
CATEGORIES = ['WGABS', 'WGALBS', 'WGBS', 'WGLBS', 'WGSBS', 'WLABS']

# Default K built from the resized image dimensions
F_DEFAULT = float(max(TARGET_H, TARGET_W))   # 640
K_DEFAULT = np.array([
    [F_DEFAULT, 0,         TARGET_W / 2],
    [0,         F_DEFAULT, TARGET_H / 2],
    [0,         0,         1           ],
], dtype=np.float64)


# ── Helpers ───────────────────────────────────────────────────────────────────

def find_image(scene_dir, stem):
    for ext in ('.jpg', '.png'):
        p = os.path.join(scene_dir, stem + ext)
        if os.path.isfile(p):
            return p
    raise FileNotFoundError(f'No {stem}.jpg/.png in {scene_dir}')


def load_gray_tensor(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (TARGET_W, TARGET_H))
    return torch.from_numpy(img.astype(np.float32) / 255.0).unsqueeze(0).unsqueeze(0)


def get_image_original_size(path):
    img = cv2.imread(path)
    return img.shape[:2]   # (H, W)


def pose_error(R_est, t_est, R_gt, t_gt):
    """Angular errors in degrees for R and t."""
    # Translation error (direction only — scale ambiguity from E decomposition)
    n = np.linalg.norm(t_est) * np.linalg.norm(t_gt)
    t_err = np.rad2deg(np.arccos(np.clip(np.dot(t_est, t_gt) / (n + 1e-8), -1.0, 1.0)))
    t_err = min(t_err, 180.0 - t_err)

    # Rotation error
    cos = (np.trace(R_est.T @ R_gt) - 1) / 2
    R_err = np.rad2deg(np.abs(np.arccos(np.clip(cos, -1.0, 1.0))))
    return R_err, t_err


def estimate_pose_from_pts(pts0, pts1, K, ransac_thr=1.0, conf=0.9999):
    """Estimate R, t from point correspondences using E decomposition."""
    if len(pts0) < 5:
        return None
    K0 = K1 = K
    # Normalize
    kpts0 = (pts0 - K[[0, 1], [2, 2]][None]) / K[[0, 1], [0, 1]][None]
    kpts1 = (pts1 - K[[0, 1], [2, 2]][None]) / K[[0, 1], [0, 1]][None]
    norm_thr = ransac_thr / K[0, 0]

    E, mask = cv2.findEssentialMat(
        kpts0, kpts1, np.eye(3),
        threshold=norm_thr, prob=conf, method=cv2.RANSAC)
    if E is None:
        return None

    best_n, best_ret = 0, None
    for _E in np.split(E, len(E) // 3):
        n, R, t, _ = cv2.recoverPose(_E, kpts0, kpts1, np.eye(3), 1e9, mask=mask)
        if n > best_n:
            best_ret = (R, t[:, 0], mask.ravel() > 0)
            best_n = n
    return best_ret


# ── Per-pair evaluation ───────────────────────────────────────────────────────

def eval_pair(model, scene_dir, device, thr=0.2):
    p01 = find_image(scene_dir, '01')
    p02 = find_image(scene_dir, '02')

    oh0, ow0 = get_image_original_size(p01)
    oh1, ow1 = get_image_original_size(p02)

    # Scale GT correspondences to resized image coords
    corrs = np.loadtxt(os.path.join(scene_dir, 'corrs.txt'))
    if corrs.ndim == 1:
        corrs = corrs.reshape(1, -1)
    gt_pts1 = corrs[:, :2] * np.array([TARGET_W / ow0, TARGET_H / oh0])
    gt_pts2 = corrs[:, 2:] * np.array([TARGET_W / ow1, TARGET_H / oh1])

    # ── GT pose from GT correspondences ──────────────────────────────────────
    gt_ret = estimate_pose_from_pts(gt_pts1.astype(np.float64),
                                    gt_pts2.astype(np.float64),
                                    K_DEFAULT, ransac_thr=1.0)
    if gt_ret is None:
        return None   # can't establish GT pose — skip pair
    R_gt, t_gt, _ = gt_ret

    # ── Predicted matches ─────────────────────────────────────────────────────
    t0 = load_gray_tensor(p01).to(device)
    t1 = load_gray_tensor(p02).to(device)
    model.matcher.coarse_matching.thr = thr
    model.matcher.coarse_matching.epipolar_F = None
    data = {'image0': t0, 'image1': t1}
    with torch.no_grad():
        model.matcher(data)
    mkpts0 = data['mkpts0_f'].cpu().numpy()
    mkpts1 = data['mkpts1_f'].cpu().numpy()
    n_pred = len(mkpts0)

    # ── Predicted pose ────────────────────────────────────────────────────────
    pred_ret = estimate_pose_from_pts(mkpts0.astype(np.float64),
                                      mkpts1.astype(np.float64),
                                      K_DEFAULT, ransac_thr=1.0)

    if pred_ret is None:
        return {
            'R_err': None, 't_err': None,
            'n_pred': n_pred, 'n_inlier': 0,
            'gt_n_pts': len(gt_pts1),
        }

    R_pred, t_pred, inliers = pred_ret
    R_err, t_err = pose_error(R_pred, t_pred, R_gt, t_gt)

    return {
        'R_err': R_err,
        't_err': t_err,
        'max_err': max(R_err, t_err),
        'n_pred': n_pred,
        'n_inlier': int(inliers.sum()),
        'gt_n_pts': len(gt_pts1),
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='/Users/siddharthraj/classes/cv/cv_final/data/WxBS_data_folder/v1.1')
    parser.add_argument('--vanilla_ckpt', default='model/weights/indoor-lite-LA.ckpt')
    parser.add_argument('--thr', type=float, default=0.2)
    parser.add_argument('--model_type', default='auto',
                        help='Model variant: auto (detect from ckpt name), indoor-lite, outdoor-large')
    args = parser.parse_args()

    if torch.cuda.is_available():
        device = 'cuda'
    elif torch.backends.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'
    print(f'Device: {device}')

    model_type = args.model_type
    if model_type == 'auto':
        ckpt_name = os.path.basename(args.vanilla_ckpt)
        model_type = 'outdoor-large' if ('outdoor' in ckpt_name and 'large' in ckpt_name) else 'indoor-lite'
    print(f'Model type: {model_type}')

    config = get_cfg_defaults()
    if model_type == 'outdoor-large':
        config.MATCHFORMER.BACKBONE_TYPE = 'largela'
        config.MATCHFORMER.SCENS = 'outdoor'
        config.MATCHFORMER.RESOLUTION = (8, 2)
        config.MATCHFORMER.COARSE.D_MODEL = 256
        config.MATCHFORMER.COARSE.D_FFN = 256
    else:
        config.MATCHFORMER.BACKBONE_TYPE = 'litela'
        config.MATCHFORMER.SCENS = 'indoor'
        config.MATCHFORMER.RESOLUTION = (8, 4)
        config.MATCHFORMER.COARSE.D_MODEL = 192
        config.MATCHFORMER.COARSE.D_FFN = 192
    model = PL_LoFTR(config, pretrained_ckpt=args.vanilla_ckpt).to(device).eval()

    # Collect scenes
    scenes = []
    for cat in CATEGORIES:
        cat_dir = os.path.join(args.data_dir, cat)
        if not os.path.isdir(cat_dir):
            continue
        for scene in sorted(os.listdir(cat_dir)):
            scene_dir = os.path.join(cat_dir, scene)
            if not os.path.isfile(os.path.join(scene_dir, 'corrs.txt')):
                continue
            img1 = next((os.path.join(scene_dir, f'01{e}') for e in ('.jpg', '.png')
                         if os.path.isfile(os.path.join(scene_dir, f'01{e}'))), None)
            img2 = next((os.path.join(scene_dir, f'02{e}') for e in ('.jpg', '.png')
                         if os.path.isfile(os.path.join(scene_dir, f'02{e}'))), None)
            if img1 and img2:
                scenes.append((cat, scene, scene_dir))

    print(f'Found {len(scenes)} image pairs\n')

    cat_results = {cat: [] for cat in CATEGORIES}
    all_results = []

    for cat, scene_name, scene_dir in tqdm(scenes, desc='Evaluating', unit='pair'):
        try:
            res = eval_pair(model, scene_dir, device, thr=args.thr)
        except Exception as e:
            print(f'\nERROR on {cat}/{scene_name}: {e}')
            continue
        if res is None:
            continue
        res['scene'] = scene_name
        cat_results[cat].append(res)
        all_results.append(res)

    AUC_THR = [5, 10, 20]

    # ── Per-scene table ───────────────────────────────────────────────────────
    print(f'\n{"="*95}')
    print('WXBS POSE BENCHMARK — Vanilla MatchFormer')
    print(f'Confidence thr={args.thr} | Assumed K: f={F_DEFAULT:.0f}, cx={TARGET_W/2:.0f}, cy={TARGET_H/2:.0f}')
    print(f'GT pose derived from GT correspondences via findEssentialMat+recoverPose')
    print(f'{"="*95}')
    print(f'{"Category":<10} {"Scene":<25} {"R_err":>7} {"t_err":>7} {"max_err":>8} {"n_pred":>7} {"n_inlier":>9}')
    print(f'{"-"*10}-{"-"*25}-{"-"*7}-{"-"*7}-{"-"*8}-{"-"*7}-{"-"*9}')

    for cat in CATEGORIES:
        for res in cat_results[cat]:
            name = res['scene']
            if res['R_err'] is not None:
                print(f'{cat:<10} {name:<25} {res["R_err"]:>6.1f}° {res["t_err"]:>6.1f}° {res["max_err"]:>7.1f}° {res["n_pred"]:>7} {res["n_inlier"]:>9}')
            else:
                print(f'{cat:<10} {name:<25} {"N/A":>7} {"N/A":>7} {"N/A":>8} {res["n_pred"]:>7} {res["n_inlier"]:>9}')

    # ── Per-category AUC ──────────────────────────────────────────────────────
    print(f'\n{"="*95}')
    print('CATEGORY AUC  (max(R_err, t_err) — lower = better)')
    print(f'{"="*95}')
    print(f'{"Category":<10} {"#valid":>7} {"AUC@5°":>9} {"AUC@10°":>9} {"AUC@20°":>9} {"med_R":>8} {"med_t":>8}')
    print(f'{"-"*10}-{"-"*7}-{"-"*9}-{"-"*9}-{"-"*9}-{"-"*8}-{"-"*8}')

    for cat in CATEGORIES:
        rlist = [r for r in cat_results[cat] if r['R_err'] is not None]
        if not rlist:
            print(f'{cat:<10} {"0":>7} {"N/A":>9} {"N/A":>9} {"N/A":>9} {"N/A":>8} {"N/A":>8}')
            continue
        max_errs = [r['max_err'] for r in rlist]
        R_errs   = [r['R_err']   for r in rlist]
        t_errs   = [r['t_err']   for r in rlist]
        aucs = error_auc(max_errs, AUC_THR)
        print(f'{cat:<10} {len(rlist):>7} '
              f'{aucs["auc@5"]*100:>8.1f}% {aucs["auc@10"]*100:>8.1f}% {aucs["auc@20"]*100:>8.1f}% '
              f'{np.median(R_errs):>7.1f}° {np.median(t_errs):>7.1f}°')

    # ── Overall ───────────────────────────────────────────────────────────────
    valid = [r for r in all_results if r['R_err'] is not None]
    print(f'\n{"="*95}')
    print('OVERALL SUMMARY')
    print(f'{"="*95}')
    if valid:
        max_errs = [r['max_err'] for r in valid]
        R_errs   = [r['R_err']   for r in valid]
        t_errs   = [r['t_err']   for r in valid]
        aucs = error_auc(max_errs, AUC_THR)
        n_no_gt = len(scenes) - len(all_results)
        print(f'  Pairs with estimable GT pose : {len(all_results)} / {len(scenes)}  ({n_no_gt} skipped — GT pts insufficient)')
        print(f'  Pairs with estimable pred pose: {len(valid)} / {len(all_results)}')
        print(f'  AUC @ 5°  : {aucs["auc@5"]*100:.1f}%')
        print(f'  AUC @ 10° : {aucs["auc@10"]*100:.1f}%')
        print(f'  AUC @ 20° : {aucs["auc@20"]*100:.1f}%')
        print(f'  Median R error : {np.median(R_errs):.1f}°')
        print(f'  Median t error : {np.median(t_errs):.1f}°')
        print(f'  Avg pred matches : {np.mean([r["n_pred"] for r in valid]):.1f}')
        print(f'  Avg RANSAC inliers: {np.mean([r["n_inlier"] for r in valid]):.1f}')
    else:
        print('  No valid results.')
    print(f'{"="*95}')


if __name__ == '__main__':
    main()
