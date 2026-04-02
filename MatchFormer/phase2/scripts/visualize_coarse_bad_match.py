"""
Visualize coarse matching for a bad prediction.

Finds a val pair from scene0000 with at least one predicted match with
reprojection error > 3px, then for one such bad match draws the top-5
coarse match candidates from the confidence matrix.

Usage:
    python visualize_coarse_bad_match.py \
        --data_dir ../data/scans/scene0000_00/exported \
        --out temp/coarse_bad_match.jpg
"""

import os
import sys
import argparse
import numpy as np
import cv2
import torch
from torch.utils.data import random_split

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config.defaultmf import get_cfg_defaults
from model.lightning_loftr import PL_LoFTR
from model.datasets.scannet_simple import ScanNetSimpleDataset

TARGET_H, TARGET_W = 480, 640
COARSE_STRIDE = 8
H_C, W_C = TARGET_H // COARSE_STRIDE, TARGET_W // COARSE_STRIDE  # 60 x 80


def get_reprojection_error(mkpts0, mkpts1, depth0, T0, T1, K):
    """Returns per-match reprojection error and validity mask."""
    fx, fy, cx, cy = K[0,0], K[1,1], K[0,2], K[1,2]
    H, W = depth0.shape

    xi = np.clip(np.round(mkpts0[:,0]).astype(int), 0, W-1)
    yi = np.clip(np.round(mkpts0[:,1]).astype(int), 0, H-1)
    z = depth0[yi, xi]
    valid = (z > 0.1) & (z <= 10.0)

    x_c = (mkpts0[:,0] - cx) * z / fx
    y_c = (mkpts0[:,1] - cy) * z / fy
    pts_h = np.stack([x_c, y_c, z, np.ones(len(mkpts0))], axis=1)

    T_0to1 = np.linalg.inv(T1) @ T0
    pts_c1 = (T_0to1 @ pts_h.T).T
    valid &= pts_c1[:,2] > 0

    u1 = np.where(valid, pts_c1[:,0] * fx / np.where(valid, pts_c1[:,2], 1) + cx, 0)
    v1 = np.where(valid, pts_c1[:,1] * fy / np.where(valid, pts_c1[:,2], 1) + cy, 0)
    valid &= (u1 >= 0) & (u1 < TARGET_W) & (v1 >= 0) & (v1 < TARGET_H)

    gt_mkpts1 = np.stack([u1, v1], axis=1)
    errs = np.where(valid, np.linalg.norm(mkpts1 - gt_mkpts1, axis=1), np.inf)
    return errs, valid, gt_mkpts1


def coarse_idx_to_pixel(idx, W_c=W_C, stride=COARSE_STRIDE):
    """Convert flat coarse index → pixel centre coordinates."""
    cy = (idx // W_c) * stride + stride // 2
    cx = (idx %  W_c) * stride + stride // 2
    return int(cx), int(cy)


def pixel_to_coarse_idx(px, py, W_c=W_C, stride=COARSE_STRIDE):
    return (py // stride) * W_c + (px // stride)


def tensor_to_bgr(t):
    """[1,1,H,W] float tensor → uint8 BGR image."""
    arr = (t[0,0].cpu().numpy() * 255).astype(np.uint8)
    return cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='../data/scans/scene0000_00/exported')
    parser.add_argument('--ckpt', default='model/weights/indoor-lite-LA.ckpt')
    parser.add_argument('--split_seed', type=int, default=42)
    parser.add_argument('--frame_gap', type=int, default=20)
    parser.add_argument('--err_thr', type=float, default=3.0, help='Min reprojection error to be "bad"')
    parser.add_argument('--top_k', type=int, default=5, help='Number of top coarse candidates to show')
    parser.add_argument('--out', default='temp/coarse_bad_match.jpg')
    args = parser.parse_args()

    device = 'mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Device: {device}')

    # ── Load model ────────────────────────────────────────────────────────────
    config = get_cfg_defaults()
    config.MATCHFORMER.BACKBONE_TYPE = 'litela'
    config.MATCHFORMER.SCENS = 'indoor'
    config.MATCHFORMER.RESOLUTION = (8, 4)
    config.MATCHFORMER.COARSE.D_MODEL = 192
    config.MATCHFORMER.COARSE.D_FFN = 192
    model = PL_LoFTR(config, pretrained_ckpt=args.ckpt).to(device).eval()

    # ── Load val split ────────────────────────────────────────────────────────
    dataset = ScanNetSimpleDataset(args.data_dir, frame_gap=args.frame_gap)
    n_val = max(1, int(len(dataset) * 0.1))
    n_train = len(dataset) - n_val
    _, val_ds = random_split(dataset, [n_train, n_val],
                             generator=torch.Generator().manual_seed(args.split_seed))
    print(f'Val pairs: {len(val_ds)}')

    # ── Find a pair with a bad match ─────────────────────────────────────────
    found = None
    for i in range(len(val_ds)):
        item = val_ds[i]
        T0 = item['T0'].numpy()
        T1 = item['T1'].numpy()
        K  = item['K'].numpy()
        depth0 = item['depth0'].numpy()
        if not np.isfinite(T0).all() or not np.isfinite(T1).all():
            continue

        img0_t = item['image0'].unsqueeze(0).to(device)
        img1_t = item['image1'].unsqueeze(0).to(device)

        data = {'image0': img0_t, 'image1': img1_t}
        model.matcher.coarse_matching.epipolar_F = None
        model.matcher.coarse_matching.thr = 0.2
        with torch.no_grad():
            model.matcher(data)

        mkpts0 = data['mkpts0_f'].cpu().numpy()
        mkpts1 = data['mkpts1_f'].cpu().numpy()
        conf_matrix = data['conf_matrix']  # [1, H0c*W0c, H1c*W1c]

        if len(mkpts0) < 1:
            continue

        errs, valid, gt_mkpts1 = get_reprojection_error(mkpts0, mkpts1, depth0, T0, T1, K)

        bad_mask = valid & (errs > args.err_thr)
        if not bad_mask.any():
            continue

        # Pick the worst valid bad match
        worst_idx = np.argmax(np.where(bad_mask, errs, -1))
        found = {
            'item': item,
            'mkpts0': mkpts0,
            'mkpts1': mkpts1,
            'errs': errs,
            'valid': valid,
            'gt_mkpts1': gt_mkpts1,
            'bad_idx': worst_idx,
            'conf_matrix': conf_matrix,
            'pair_idx': i,
        }
        print(f'Found pair {i} with bad match (err={errs[worst_idx]:.1f}px)')
        break

    if found is None:
        print('No bad match found in val set.')
        return

    # ── Extract info ──────────────────────────────────────────────────────────
    bad_idx = found['bad_idx']
    bad_pt0 = found['mkpts0'][bad_idx]        # fine match in image0
    bad_pt1 = found['mkpts1'][bad_idx]        # predicted fine match in image1
    gt_pt1  = found['gt_mkpts1'][bad_idx]     # GT location in image1
    err     = found['errs'][bad_idx]
    conf_matrix = found['conf_matrix']         # [1, 4800, 4800]

    # Coarse index of the query point in image0
    q_coarse_idx = pixel_to_coarse_idx(int(bad_pt0[0]), int(bad_pt0[1]))

    # Top-K coarse matches for this query in image1
    conf_row = conf_matrix[0, q_coarse_idx, :].cpu().numpy()  # [H1c*W1c]
    topk_flat = np.argsort(conf_row)[::-1][:args.top_k]
    topk_confs = conf_row[topk_flat]

    print(f'\nQuery point (image0): ({bad_pt0[0]:.1f}, {bad_pt0[1]:.1f})')
    print(f'Coarse grid cell: ({int(bad_pt0[0])//8}, {int(bad_pt0[1])//8})  →  flat idx {q_coarse_idx}')
    print(f'Predicted fine match (image1): ({bad_pt1[0]:.1f}, {bad_pt1[1]:.1f})')
    print(f'GT location (image1):          ({gt_pt1[0]:.1f}, {gt_pt1[1]:.1f})')
    print(f'Reprojection error: {err:.1f}px')
    print(f'\nTop-{args.top_k} coarse candidates in image1:')
    for rank, (flat_idx, conf) in enumerate(zip(topk_flat, topk_confs)):
        px, py = coarse_idx_to_pixel(flat_idx)
        print(f'  #{rank+1}: coarse_idx={flat_idx}  pixel=({px},{py})  conf={conf:.4f}')

    # ── Visualise ─────────────────────────────────────────────────────────────
    item = found['item']
    img0_bgr = tensor_to_bgr(item['image0'].unsqueeze(0))
    img1_bgr = tensor_to_bgr(item['image1'].unsqueeze(0))

    PAD = 12
    canvas = np.ones((TARGET_H, TARGET_W*2 + PAD, 3), dtype=np.uint8) * 30
    canvas[:, :TARGET_W] = img0_bgr
    canvas[:, TARGET_W+PAD:] = img1_bgr

    def pt0(p): return (int(p[0]),           int(p[1]))
    def pt1(p): return (int(p[0])+TARGET_W+PAD, int(p[1]))

    # Draw the coarse grid cell highlight on image0
    cx0 = (int(bad_pt0[0]) // 8) * 8
    cy0 = (int(bad_pt0[1]) // 8) * 8
    cv2.rectangle(canvas, (cx0, cy0), (cx0+8, cy0+8), (255, 255, 0), 1)

    # Query point in image0 (yellow)
    cv2.circle(canvas, pt0(bad_pt0), 6, (0, 255, 255), -1)
    cv2.circle(canvas, pt0(bad_pt0), 6, (0, 0, 0), 1)

    # Top-K coarse candidates in image1 (colour-coded by rank)
    colors = [
        (0,   255,   0),   # #1 green
        (0,   200,  80),   # #2
        (0,   140, 160),   # #3
        (0,    80, 220),   # #4
        (0,     0, 255),   # #5 blue
    ]
    for rank, (flat_idx, conf) in enumerate(zip(topk_flat, topk_confs)):
        px, py = coarse_idx_to_pixel(flat_idx)
        col = colors[rank]
        # Coarse cell rectangle on image1
        rx = (flat_idx % W_C) * COARSE_STRIDE
        ry = (flat_idx // W_C) * COARSE_STRIDE
        cv2.rectangle(canvas, pt1((rx, ry)), pt1((rx+8, ry+8)), col, 2)
        # Centre dot
        cv2.circle(canvas, pt1((px, py)), 5, col, -1)
        # Rank label
        cv2.putText(canvas, f'#{rank+1} {conf:.3f}',
                    (pt1((px, py))[0]+6, pt1((px, py))[1]+4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, col, 1, cv2.LINE_AA)

    # Predicted fine match in image1 (red cross)
    cv2.drawMarker(canvas, pt1(bad_pt1), (0, 0, 255), cv2.MARKER_CROSS, 14, 2)
    cv2.putText(canvas, f'pred ({bad_pt1[0]:.0f},{bad_pt1[1]:.0f})',
                (pt1(bad_pt1)[0]+8, pt1(bad_pt1)[1]-6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0,0,255), 1, cv2.LINE_AA)

    # GT location in image1 (magenta cross)
    cv2.drawMarker(canvas, pt1(gt_pt1), (255, 0, 255), cv2.MARKER_CROSS, 14, 2)
    cv2.putText(canvas, f'GT ({gt_pt1[0]:.0f},{gt_pt1[1]:.0f})',
                (pt1(gt_pt1)[0]+8, pt1(gt_pt1)[1]+12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255,0,255), 1, cv2.LINE_AA)

    # Line from pred to GT (dashed effect via dotted line)
    cv2.line(canvas, pt1(bad_pt1), pt1(gt_pt1), (100, 100, 255), 1, cv2.LINE_AA)

    # Legend & title
    cv2.putText(canvas, f'Query (img0): ({bad_pt0[0]:.0f},{bad_pt0[1]:.0f})  |  pair={found["pair_idx"]}',
                (8, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220,220,220), 1)
    cv2.putText(canvas, f'Reprojection err: {err:.1f}px  |  cyan=query  red=pred  magenta=GT  green..blue=top-{args.top_k} coarse',
                (8, TARGET_H-8), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200,200,200), 1)

    os.makedirs(os.path.dirname(args.out) if os.path.dirname(args.out) else '.', exist_ok=True)
    cv2.imwrite(args.out, canvas)
    print(f'\nSaved: {args.out}')


if __name__ == '__main__':
    main()
