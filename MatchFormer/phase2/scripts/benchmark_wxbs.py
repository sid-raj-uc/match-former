"""
WxBS Benchmark — Vanilla MatchFormer.

For each image pair in the WxBS dataset:
  1. Run vanilla MatchFormer to get predicted matches.
  2. Estimate Fundamental matrix via RANSAC from predictions.
  3. Measure symmetric epipolar distance of GT correspondences w.r.t. estimated F.

Metrics reported per category and overall:
  - med_epi   : median symmetric epipolar error on GT points (lower = better)
  - mean_epi  : mean symmetric epipolar error on GT points
  - AUC@1/3/5 : area under the cumulative error curve at 1/3/5px thresholds
  - @1px / @3px / @5px : % of GT correspondences within threshold
  - n_pred    : number of predicted matches
  - n_inlier  : number of RANSAC inliers

Usage:
    python benchmark_wxbs.py --data_dir /path/to/WxBS_data_folder/v1.1
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

TARGET_H, TARGET_W = 480, 640
CATEGORIES = ['WGABS', 'WGALBS', 'WGBS', 'WGLBS', 'WGSBS', 'WLABS']


# ── Image loading ─────────────────────────────────────────────────────────────

def find_image(scene_dir, stem):
    """Find image file with given stem (e.g. '01') trying .jpg then .png."""
    for ext in ('.jpg', '.png'):
        p = os.path.join(scene_dir, stem + ext)
        if os.path.isfile(p):
            return p
    raise FileNotFoundError(f'No image {stem}.jpg/.png in {scene_dir}')


def load_image_tensor(path):
    """Load an image, convert to grayscale, resize to TARGET_HxW, return [1,1,H,W] float32 tensor."""
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f'Cannot read {path}')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (TARGET_W, TARGET_H))
    tensor = torch.from_numpy(gray.astype(np.float32) / 255.0).unsqueeze(0).unsqueeze(0)
    return tensor


# ── Symmetric epipolar distance ───────────────────────────────────────────────

def sym_epipolar_dist(F, pts1, pts2):
    """
    Compute symmetric epipolar distance for N GT correspondence pairs.

    Args:
        F   : [3,3] fundamental matrix  (pts2^T F pts1 = 0)
        pts1: [N,2]  points in image 1
        pts2: [N,2]  points in image 2
    Returns:
        dists: [N] float32 distances in pixels
    """
    n = len(pts1)
    ones = np.ones((n, 1), dtype=np.float64)
    p1 = np.hstack([pts1, ones])   # [N,3]
    p2 = np.hstack([pts2, ones])   # [N,3]

    Fp1  = (F @ p1.T).T           # [N,3]  epiline in image 2
    FTp2 = (F.T @ p2.T).T         # [N,3]  epiline in image 1

    numer = np.abs(np.sum(p2 * Fp1, axis=1))  # |p2^T F p1|  (same as |p1^T F^T p2|)

    # Point-to-epipolar-line distances (in pixels)
    d1 = numer / (np.sqrt(Fp1[:, 0]**2  + Fp1[:, 1]**2)  + 1e-8)  # dist p2 → line Fp1
    d2 = numer / (np.sqrt(FTp2[:, 0]**2 + FTp2[:, 1]**2) + 1e-8)  # dist p1 → line F^T p2

    # Symmetric epipolar distance: sum of both distances (pixels)
    return (d1 + d2).astype(np.float32)


# ── Model inference ───────────────────────────────────────────────────────────

def run_model(model, img0_tensor, img1_tensor, device, thr=0.2):
    """Run MatchFormer on a pair of image tensors."""
    model.matcher.coarse_matching.thr = thr
    model.matcher.coarse_matching.epipolar_F = None   # vanilla: no epipolar constraint
    data = {
        'image0': img0_tensor.to(device),
        'image1': img1_tensor.to(device),
    }
    with torch.no_grad():
        model.matcher(data)
    mkpts0 = data['mkpts0_f'].cpu().numpy()
    mkpts1 = data['mkpts1_f'].cpu().numpy()
    return mkpts0, mkpts1


# ── Per-pair evaluation ───────────────────────────────────────────────────────

def eval_pair(model, scene_dir, device, thr=0.2, ransac_thresh=1.0):
    p01 = find_image(scene_dir, '01')
    p02 = find_image(scene_dir, '02')
    img0_t = load_image_tensor(p01)
    img1_t = load_image_tensor(p02)

    # Load GT correspondences (x1 y1 x2 y2)
    corrs = np.loadtxt(os.path.join(scene_dir, 'corrs.txt'))
    if corrs.ndim == 1:
        corrs = corrs.reshape(1, -1)
    gt_pts1 = corrs[:, :2].astype(np.float64)
    gt_pts2 = corrs[:, 2:].astype(np.float64)

    # Scale GT coords from original image size → resized size
    orig  = cv2.imread(p01)
    orig_h,  orig_w  = orig.shape[:2]
    gt_pts1_scaled = gt_pts1 * np.array([TARGET_W / orig_w,  TARGET_H / orig_h])

    orig2 = cv2.imread(p02)
    orig_h2, orig_w2 = orig2.shape[:2]
    gt_pts2_scaled = gt_pts2 * np.array([TARGET_W / orig_w2, TARGET_H / orig_h2])

    # Run model
    mkpts0, mkpts1 = run_model(model, img0_t, img1_t, device, thr=thr)
    n_pred = len(mkpts0)

    if n_pred < 8:
        # Not enough matches to estimate F
        return {
            'n_pred': n_pred,
            'n_inlier': 0,
            'epi_errors': None,
        }

    # Estimate F via RANSAC
    F_est, mask = cv2.findFundamentalMat(
        mkpts0.astype(np.float64),
        mkpts1.astype(np.float64),
        cv2.FM_RANSAC,
        ransacReprojThreshold=ransac_thresh,
        confidence=0.999,
        maxIters=10000,
    )
    n_inlier = int(mask.sum()) if mask is not None else 0

    if F_est is None or not np.isfinite(F_est).all():
        return {
            'n_pred': n_pred,
            'n_inlier': n_inlier,
            'epi_errors': None,
        }

    # Symmetric epipolar distance on GT points
    epi_errors = sym_epipolar_dist(F_est, gt_pts1_scaled, gt_pts2_scaled)

    return {
        'n_pred': n_pred,
        'n_inlier': n_inlier,
        'epi_errors': epi_errors,
    }


# ── AUC helper ────────────────────────────────────────────────────────────────

def auc_at(errors, thresholds):
    """Compute AUC of cumulative-error curve at given thresholds (trapezoidal)."""
    if len(errors) == 0:
        return {t: 0.0 for t in thresholds}
    errors_sorted = np.sort(errors)
    n = len(errors_sorted)
    cum = np.arange(1, n + 1) / n
    result = {}
    for thr in thresholds:
        # sample cumulative proportion at thr
        pct = float(np.mean(errors_sorted <= thr))
        result[thr] = pct * 100.0  # treat as % under curve (area up to thr / thr)
    return result


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='WxBS benchmark for vanilla MatchFormer')
    parser.add_argument('--data_dir', default='/Users/siddharthraj/classes/cv/cv_final/data/WxBS_data_folder/v1.1')
    parser.add_argument('--vanilla_ckpt', default='model/weights/indoor-lite-LA.ckpt')
    parser.add_argument('--thr', type=float, default=0.2, help='MatchFormer confidence threshold')
    parser.add_argument('--ransac_thresh', type=float, default=1.0, help='RANSAC reprojection threshold (px)')
    parser.add_argument('--model_type', default='auto',
                        help='Model variant: auto (detect from ckpt name), indoor-lite, outdoor-large')
    args = parser.parse_args()

    # Device
    if torch.cuda.is_available():
        device = 'cuda'
    elif torch.backends.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'
    print(f'Device: {device}')

    # Detect model type from checkpoint name if auto
    model_type = args.model_type
    if model_type == 'auto':
        ckpt_name = os.path.basename(args.vanilla_ckpt)
        if 'outdoor' in ckpt_name and 'large' in ckpt_name:
            model_type = 'outdoor-large'
        else:
            model_type = 'indoor-lite'
    print(f'Model type: {model_type}')

    # Load model
    config = get_cfg_defaults()
    if model_type == 'outdoor-large':
        config.MATCHFORMER.BACKBONE_TYPE = 'largela'
        config.MATCHFORMER.SCENS = 'outdoor'
        config.MATCHFORMER.RESOLUTION = (8, 2)
        config.MATCHFORMER.COARSE.D_MODEL = 256
        config.MATCHFORMER.COARSE.D_FFN = 256
    else:  # indoor-lite
        config.MATCHFORMER.BACKBONE_TYPE = 'litela'
        config.MATCHFORMER.SCENS = 'indoor'
        config.MATCHFORMER.RESOLUTION = (8, 4)
        config.MATCHFORMER.COARSE.D_MODEL = 192
        config.MATCHFORMER.COARSE.D_FFN = 192

    print(f'Loading vanilla model from {args.vanilla_ckpt} ...')
    model = PL_LoFTR(config, pretrained_ckpt=args.vanilla_ckpt).to(device).eval()

    # Collect all scenes
    scenes = []  # list of (category, scene_name, scene_dir)
    for cat in CATEGORIES:
        cat_dir = os.path.join(args.data_dir, cat)
        if not os.path.isdir(cat_dir):
            continue
        for scene in sorted(os.listdir(cat_dir)):
            scene_dir = os.path.join(cat_dir, scene)
            corrs_file = os.path.join(scene_dir, 'corrs.txt')
            if not os.path.isfile(corrs_file):
                continue
            # Accept either .jpg or .png
            img1 = next((os.path.join(scene_dir, f'01{ext}') for ext in ('.jpg', '.png')
                         if os.path.isfile(os.path.join(scene_dir, f'01{ext}'))), None)
            img2 = next((os.path.join(scene_dir, f'02{ext}') for ext in ('.jpg', '.png')
                         if os.path.isfile(os.path.join(scene_dir, f'02{ext}'))), None)
            if img1 and img2:
                scenes.append((cat, scene, scene_dir))

    print(f'Found {len(scenes)} image pairs across {len(CATEGORIES)} categories\n')

    # Evaluate
    cat_results = {cat: [] for cat in CATEGORIES}
    all_epi = []

    pbar = tqdm(scenes, desc='Evaluating', unit='pair')
    for cat, scene_name, scene_dir in pbar:
        pbar.set_postfix({'scene': f'{cat}/{scene_name}'})
        try:
            res = eval_pair(model, scene_dir, device, thr=args.thr, ransac_thresh=args.ransac_thresh)
        except Exception as e:
            print(f'\nERROR on {cat}/{scene_name}: {e}')
            continue
        res['scene'] = scene_name
        cat_results[cat].append(res)
        if res['epi_errors'] is not None:
            all_epi.append(res['epi_errors'])

    # ── Per-scene table ───────────────────────────────────────────────────────
    THR = [1.0, 3.0, 5.0]
    print(f'\n{"="*90}')
    print('WXBS BENCHMARK — Vanilla MatchFormer')
    print(f'Confidence threshold: {args.thr}  |  RANSAC threshold: {args.ransac_thresh}px')
    print(f'{"="*90}')
    print(f'{"Category":<10} {"Scene":<25} {"med_epi":>8} {"mean_epi":>9} {"@1px":>7} {"@3px":>7} {"@5px":>7} {"n_pred":>7} {"n_inlier":>9}')
    print(f'{"-"*10}-{"-"*25}-{"-"*8}-{"-"*9}-{"-"*7}-{"-"*7}-{"-"*7}-{"-"*7}-{"-"*9}')

    for cat in CATEGORIES:
        for res in cat_results[cat]:
            name = res['scene']
            n_pred = res['n_pred']
            n_inlier = res['n_inlier']
            errs = res['epi_errors']
            if errs is not None and len(errs) > 0:
                med_e = np.median(errs)
                mean_e = np.mean(errs)
                p1  = np.mean(errs <= 1.0) * 100
                p3  = np.mean(errs <= 3.0) * 100
                p5  = np.mean(errs <= 5.0) * 100
                print(f'{cat:<10} {name:<25} {med_e:>8.3f} {mean_e:>9.3f} {p1:>6.1f}% {p3:>6.1f}% {p5:>6.1f}% {n_pred:>7} {n_inlier:>9}')
            else:
                print(f'{cat:<10} {name:<25} {"N/A":>8} {"N/A":>9} {"N/A":>7} {"N/A":>7} {"N/A":>7} {n_pred:>7} {n_inlier:>9}')

    # ── Per-category summary ──────────────────────────────────────────────────
    print(f'\n{"="*90}')
    print('CATEGORY SUMMARY')
    print(f'{"="*90}')
    print(f'{"Category":<10} {"#pairs":>7} {"med_epi":>8} {"mean_epi":>9} {"@1px":>7} {"@3px":>7} {"@5px":>7} {"avg_pred":>9} {"avg_inlier":>11}')
    print(f'{"-"*10}-{"-"*7}-{"-"*8}-{"-"*9}-{"-"*7}-{"-"*7}-{"-"*7}-{"-"*9}-{"-"*11}')

    for cat in CATEGORIES:
        rlist = cat_results[cat]
        if not rlist:
            continue
        cat_errs = [r['epi_errors'] for r in rlist if r['epi_errors'] is not None]
        all_cat_errs = np.concatenate(cat_errs) if cat_errs else np.array([])
        avg_pred = np.mean([r['n_pred'] for r in rlist])
        avg_inlier = np.mean([r['n_inlier'] for r in rlist])
        n_pairs = len(rlist)
        if len(all_cat_errs) > 0:
            med_e  = np.median(all_cat_errs)
            mean_e = np.mean(all_cat_errs)
            p1  = np.mean(all_cat_errs <= 1.0) * 100
            p3  = np.mean(all_cat_errs <= 3.0) * 100
            p5  = np.mean(all_cat_errs <= 5.0) * 100
            print(f'{cat:<10} {n_pairs:>7} {med_e:>8.3f} {mean_e:>9.3f} {p1:>6.1f}% {p3:>6.1f}% {p5:>6.1f}% {avg_pred:>9.1f} {avg_inlier:>11.1f}')
        else:
            print(f'{cat:<10} {n_pairs:>7} {"N/A":>8} {"N/A":>9} {"N/A":>7} {"N/A":>7} {"N/A":>7} {avg_pred:>9.1f} {avg_inlier:>11.1f}')

    # ── Overall summary ───────────────────────────────────────────────────────
    print(f'\n{"="*90}')
    print('OVERALL SUMMARY')
    print(f'{"="*90}')
    all_concat = np.concatenate(all_epi) if all_epi else np.array([])
    all_preds  = sum(r['n_pred'] for rlist in cat_results.values() for r in rlist)
    all_inlier = sum(r['n_inlier'] for rlist in cat_results.values() for r in rlist)
    total_pairs = sum(len(v) for v in cat_results.values())

    if len(all_concat) > 0:
        print(f'  Pairs evaluated : {total_pairs}')
        print(f'  GT points total : {len(all_concat)}')
        print(f'  Median epi error: {np.median(all_concat):.3f} px')
        print(f'  Mean epi error  : {np.mean(all_concat):.3f} px')
        print(f'  @ 1px           : {np.mean(all_concat <= 1.0)*100:.1f}%')
        print(f'  @ 3px           : {np.mean(all_concat <= 3.0)*100:.1f}%')
        print(f'  @ 5px           : {np.mean(all_concat <= 5.0)*100:.1f}%')
        print(f'  Avg pred matches: {all_preds/total_pairs:.1f}')
        print(f'  Avg RANSAC inliers: {all_inlier/total_pairs:.1f}')
    else:
        print('  No valid results.')
    print(f'{"="*90}')


if __name__ == '__main__':
    main()
