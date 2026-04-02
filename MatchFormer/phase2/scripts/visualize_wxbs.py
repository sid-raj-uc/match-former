"""
Visualize vanilla MatchFormer predictions on WxBS pairs.

For each selected scene, produces a side-by-side image showing:
  - Predicted matches (colored by correctness: green ≤ threshold, red > threshold)
  - GT correspondences (cyan dots)
  - Epipolar lines for GT points drawn on both images

Usage:
    python visualize_wxbs.py --data_dir /path/to/WxBS_data_folder/v1.1 --out_dir temp/wxbs_vis
"""

import os
import sys
import argparse
import numpy as np
import cv2
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config.defaultmf import get_cfg_defaults
from model.lightning_loftr import PL_LoFTR

TARGET_H, TARGET_W = 480, 640

SCENES_TO_VIS = [
    ('WLABS',  'kpi'),           # best case
    ('WGALBS', 'kyiv_dolltheater'),  # medium
    ('WGLBS',  'warsaw'),        # hard / high error
]


def find_image(scene_dir, stem):
    for ext in ('.jpg', '.png'):
        p = os.path.join(scene_dir, stem + ext)
        if os.path.isfile(p):
            return p
    raise FileNotFoundError(f'No {stem}.jpg/.png in {scene_dir}')


def load_bgr_resized(path):
    img = cv2.imread(path)
    orig_h, orig_w = img.shape[:2]
    img_resized = cv2.resize(img, (TARGET_W, TARGET_H))
    return img_resized, orig_h, orig_w


def load_gray_tensor(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (TARGET_W, TARGET_H))
    return torch.from_numpy(img.astype(np.float32) / 255.0).unsqueeze(0).unsqueeze(0)


def sym_epipolar_dist(F, pts1, pts2):
    n = len(pts1)
    ones = np.ones((n, 1), dtype=np.float64)
    p1 = np.hstack([pts1, ones])
    p2 = np.hstack([pts2, ones])
    Fp1  = (F @ p1.T).T
    FTp2 = (F.T @ p2.T).T
    numer = np.abs(np.sum(p2 * Fp1, axis=1))
    d1 = numer / (np.sqrt(Fp1[:, 0]**2  + Fp1[:, 1]**2)  + 1e-8)
    d2 = numer / (np.sqrt(FTp2[:, 0]**2 + FTp2[:, 1]**2) + 1e-8)
    return (d1 + d2).astype(np.float32)


def draw_epipolar_line(img, l, color=(255, 200, 0), thickness=1):
    """Draw epipolar line l=[a,b,c] (ax+by+c=0) clipped to image bounds."""
    h, w = img.shape[:2]
    a, b, c = l[0], l[1], l[2]
    pts = []
    if abs(b) > 1e-6:
        x0, x1 = 0, w - 1
        y0 = int(-(a * x0 + c) / b)
        y1 = int(-(a * x1 + c) / b)
        pts = [(x0, y0), (x1, y1)]
    elif abs(a) > 1e-6:
        y0, y1 = 0, h - 1
        x0 = int(-(b * y0 + c) / a)
        x1 = int(-(b * y1 + c) / a)
        pts = [(x0, y0), (x1, y1)]
    if len(pts) == 2:
        cv2.line(img, pts[0], pts[1], color, thickness, cv2.LINE_AA)


def visualize_pair(model, scene_dir, device, out_path, thr_correct=5.0):
    p01 = find_image(scene_dir, '01')
    p02 = find_image(scene_dir, '02')

    img0_bgr, oh0, ow0 = load_bgr_resized(p01)
    img1_bgr, oh1, ow1 = load_bgr_resized(p02)

    corrs = np.loadtxt(os.path.join(scene_dir, 'corrs.txt'))
    if corrs.ndim == 1:
        corrs = corrs.reshape(1, -1)
    gt_pts1 = corrs[:, :2] * np.array([TARGET_W / ow0, TARGET_H / oh0])
    gt_pts2 = corrs[:, 2:] * np.array([TARGET_W / ow1, TARGET_H / oh1])

    # Run model
    t0 = load_gray_tensor(p01).to(device)
    t1 = load_gray_tensor(p02).to(device)
    model.matcher.coarse_matching.thr = 0.2
    model.matcher.coarse_matching.epipolar_F = None
    data = {'image0': t0, 'image1': t1}
    with torch.no_grad():
        model.matcher(data)
    mkpts0 = data['mkpts0_f'].cpu().numpy()
    mkpts1 = data['mkpts1_f'].cpu().numpy()
    n_pred = len(mkpts0)

    # Estimate F
    F_est = None
    inlier_mask = None
    if n_pred >= 8:
        F_est, mask = cv2.findFundamentalMat(
            mkpts0.astype(np.float64), mkpts1.astype(np.float64),
            cv2.FM_RANSAC, ransacReprojThreshold=1.0, confidence=0.999, maxIters=10000)
        if F_est is not None and np.isfinite(F_est).all():
            inlier_mask = mask.ravel().astype(bool)

    # ── Build canvas ─────────────────────────────────────────────────────────
    pad = 10
    canvas = np.ones((TARGET_H, TARGET_W * 2 + pad, 3), dtype=np.uint8) * 40
    canvas[:, :TARGET_W] = img0_bgr
    canvas[:, TARGET_W + pad:] = img1_bgr

    def pt0(p): return (int(p[0]), int(p[1]))
    def pt1(p): return (int(p[0]) + TARGET_W + pad, int(p[1]))

    # Draw epipolar lines for GT points (using estimated F)
    if F_est is not None:
        ones = np.ones((len(gt_pts1), 1))
        p1h = np.hstack([gt_pts1, ones])
        p2h = np.hstack([gt_pts2, ones])
        lines_in_img2 = (F_est @ p1h.T).T   # Fp1 → line in img2
        lines_in_img1 = (F_est.T @ p2h.T).T  # F^T p2 → line in img1
        img0_with_epi = canvas[:, :TARGET_W].copy()
        img1_with_epi = canvas[:, TARGET_W + pad:].copy()
        for l in lines_in_img1:
            draw_epipolar_line(img0_with_epi, l, color=(200, 180, 0), thickness=1)
        for l in lines_in_img2:
            draw_epipolar_line(img1_with_epi, l, color=(200, 180, 0), thickness=1)
        canvas[:, :TARGET_W] = img0_with_epi
        canvas[:, TARGET_W + pad:] = img1_with_epi

    # Draw GT correspondences (cyan)
    for i in range(len(gt_pts1)):
        cv2.circle(canvas, pt0(gt_pts1[i]), 4, (255, 220, 0), -1)
        cv2.circle(canvas, pt1(gt_pts2[i]), 4, (255, 220, 0), -1)

    # Draw predicted matches — color by epipolar error if F available
    if n_pred > 0:
        if F_est is not None:
            epi_errs = sym_epipolar_dist(F_est, mkpts0, mkpts1)
            for i, (p0, p1_) in enumerate(zip(mkpts0, mkpts1)):
                is_inlier = inlier_mask[i] if inlier_mask is not None else False
                color = (0, 220, 0) if is_inlier else (0, 0, 220)
                cv2.line(canvas, pt0(p0), pt1(p1_), color, 1, cv2.LINE_AA)
                cv2.circle(canvas, pt0(p0), 3, color, -1)
                cv2.circle(canvas, pt1(p1_), 3, color, -1)
        else:
            for p0, p1_ in zip(mkpts0, mkpts1):
                cv2.line(canvas, pt0(p0), pt1(p1_), (128, 128, 128), 1, cv2.LINE_AA)
                cv2.circle(canvas, pt0(p0), 3, (128, 128, 128), -1)
                cv2.circle(canvas, pt1(p1_), 3, (128, 128, 128), -1)

    # Stats text
    n_inlier = int(inlier_mask.sum()) if inlier_mask is not None else 0
    if F_est is not None:
        epi_errs = sym_epipolar_dist(F_est, gt_pts1, gt_pts2)
        med_e = np.median(epi_errs)
        p3 = np.mean(epi_errs <= 3.0) * 100
        stats = f'pred={n_pred} inliers={n_inlier} | GT med_epi={med_e:.1f}px @3px={p3:.0f}%'
    else:
        stats = f'pred={n_pred} | F estimation failed (need >=8 matches)'

    cv2.putText(canvas, stats, (10, TARGET_H - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    # Legend
    cv2.putText(canvas, 'green=inlier  red=outlier  cyan=GT  yellow=epi-line',
                (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)

    cv2.imwrite(out_path, canvas)
    print(f'  Saved: {out_path}  ({stats})')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='/Users/siddharthraj/classes/cv/cv_final/data/WxBS_data_folder/v1.1')
    parser.add_argument('--vanilla_ckpt', default='model/weights/indoor-lite-LA.ckpt')
    parser.add_argument('--out_dir', default='temp/wxbs_vis')
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    if torch.cuda.is_available():
        device = 'cuda'
    elif torch.backends.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'
    print(f'Device: {device}')

    config = get_cfg_defaults()
    config.MATCHFORMER.BACKBONE_TYPE = 'litela'
    config.MATCHFORMER.SCENS = 'indoor'
    config.MATCHFORMER.RESOLUTION = (8, 4)
    config.MATCHFORMER.COARSE.D_MODEL = 192
    config.MATCHFORMER.COARSE.D_FFN = 192
    model = PL_LoFTR(config, pretrained_ckpt=args.vanilla_ckpt).to(device).eval()

    for cat, scene in SCENES_TO_VIS:
        scene_dir = os.path.join(args.data_dir, cat, scene)
        if not os.path.isdir(scene_dir):
            print(f'Skipping {cat}/{scene} — not found')
            continue
        out_path = os.path.join(args.out_dir, f'{cat}_{scene}.jpg')
        print(f'\n{cat}/{scene}')
        visualize_pair(model, scene_dir, device, out_path)

    print('\nDone.')


if __name__ == '__main__':
    main()
