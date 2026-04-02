"""
Visualize top-K coarse matches for selected query points.

For each query point in image 0, draws the top-K highest confidence
matches from the coarse confidence matrix onto image 1.
"""

import os
import sys
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config.defaultmf import get_cfg_defaults
from model.lightning_loftr import PL_LoFTR

# ── Settings ─────────────────────────────────────────────────────────────────
DATA_DIR   = '../data/scans/scene0000_00/exported'
CKPT_PATH  = 'model/weights/indoor-lite-LA.ckpt'   # pretrained
TOP_K      = 5
H_IMG, W_IMG = 480, 640
STRIDE     = 8
H_C, W_C   = H_IMG // STRIDE, W_IMG // STRIDE  # 60, 80

# Query points in pixel coordinates (x, y) — pick a few interesting locations
QUERY_POINTS = [
    (320, 240),  # center
    (160, 120),  # upper-left area
    (480, 360),  # lower-right area
]

PAIR_INDICES = [0, 10, 30]   # which frame pairs to visualize
FRAME_GAP   = 20

# Colors for top-K matches (best → worst)
COLORS = [
    (0, 255, 0),    # green  — rank 1
    (0, 200, 255),  # cyan   — rank 2
    (255, 200, 0),  # yellow — rank 3
    (255, 100, 0),  # orange — rank 4
    (255, 0, 0),    # red    — rank 5
]


def build_model(ckpt_path, device):
    config = get_cfg_defaults()
    config.MATCHFORMER.BACKBONE_TYPE = 'litela'
    config.MATCHFORMER.SCENS = 'indoor'
    config.MATCHFORMER.RESOLUTION = (8, 4)
    config.MATCHFORMER.COARSE.D_MODEL = 192
    config.MATCHFORMER.COARSE.D_FFN = 192
    model = PL_LoFTR(config, pretrained_ckpt=ckpt_path)
    model.to(device).eval()
    return model


def pixel_to_coarse(px, py):
    """Convert pixel coords to coarse grid linear index."""
    cx = int(px / STRIDE)
    cy = int(py / STRIDE)
    cx = min(cx, W_C - 1)
    cy = min(cy, H_C - 1)
    return cy * W_C + cx


def coarse_to_pixel(linear_idx):
    """Convert coarse grid linear index to pixel center coords."""
    cx = linear_idx % W_C
    cy = linear_idx // W_C
    px = cx * STRIDE + STRIDE // 2
    py = cy * STRIDE + STRIDE // 2
    return px, py


def load_pair(data_dir, idx, frame_gap=20):
    """Load an image pair."""
    color_dir = os.path.join(data_dir, 'color')
    frames = sorted(
        [f for f in os.listdir(color_dir) if f.endswith('.jpg')],
        key=lambda x: int(x.split('.')[0])
    )
    f0 = frames[idx]
    f1 = frames[idx + frame_gap]

    img0 = cv2.imread(os.path.join(color_dir, f0))
    img1 = cv2.imread(os.path.join(color_dir, f1))
    img0 = cv2.resize(img0, (W_IMG, H_IMG))
    img1 = cv2.resize(img1, (W_IMG, H_IMG))

    gray0 = cv2.cvtColor(img0, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0

    # Load poses and calculate Fundamental matrix
    pose_dir = os.path.join(data_dir, 'pose')
    idx0_num = f0.split('.')[0]
    idx1_num = f1.split('.')[0]
    try:
        T0 = np.loadtxt(os.path.join(pose_dir, f'{idx0_num}.txt'))
        T1 = np.loadtxt(os.path.join(pose_dir, f'{idx1_num}.txt'))
        K = np.loadtxt(os.path.join(data_dir, 'intrinsic', 'intrinsic_depth.txt'))[:3, :3]
        from gt_epipolar import compute_fundamental_matrix
        F = compute_fundamental_matrix(T0, T1, K, K)
    except Exception as e:
        print(f"Warning: Could not compute F matrix ({e})")
        F = None

    return img0, img1, gray0, gray1, F


def visualize_pair(img0, img1, conf_matrix, query_points, top_k, pair_label, F=None):
    """
    Draw query points on img0 and their top-K matches on img1.
    conf_matrix: [L, S] numpy array (squeezed from batch dim).
    """
    n_queries = len(query_points)
    fig, axes = plt.subplots(n_queries, 2, figsize=(16, 5 * n_queries))
    if n_queries == 1:
        axes = axes[np.newaxis, :]

    for q_idx, (qx, qy) in enumerate(query_points):
        # Get coarse index for query point
        q_linear = pixel_to_coarse(qx, qy)
        row = conf_matrix[q_linear, :]  # [S] — confidence for all locations in img1

        # Top-K matches
        top_indices = np.argsort(row)[::-1][:top_k]
        top_confs = row[top_indices]

        # Draw image 0 with query point
        vis0 = img0.copy()
        cv2.circle(vis0, (qx, qy), 10, (0, 0, 255), 3)
        cv2.circle(vis0, (qx, qy), 3, (0, 0, 255), -1)
        # Draw coarse grid cell
        cx, cy = int(qx / STRIDE) * STRIDE, int(qy / STRIDE) * STRIDE
        cv2.rectangle(vis0, (cx, cy), (cx + STRIDE, cy + STRIDE), (0, 0, 255), 2)

        # Draw image 1 with top-K matches
        vis1 = img1.copy()
        for rank, (match_idx, conf) in enumerate(zip(top_indices, top_confs)):
            mx, my = coarse_to_pixel(match_idx)
            color = COLORS[rank % len(COLORS)]
            # Draw grid cell
            mcx, mcy = (mx - STRIDE // 2), (my - STRIDE // 2)
            cv2.rectangle(vis1, (mcx, mcy), (mcx + STRIDE, mcy + STRIDE), color, 2)
            cv2.circle(vis1, (mx, my), 6, color, -1)
            # Label with rank and confidence
            label = f"#{rank+1} {conf:.4f}"
            cv2.putText(vis1, label, (mx + 10, my - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

        axes[q_idx, 0].imshow(cv2.cvtColor(vis0, cv2.COLOR_BGR2RGB))
        axes[q_idx, 0].set_title(f'Image 0 — Query ({qx}, {qy})', fontsize=12)
        axes[q_idx, 0].axis('off')

        axes[q_idx, 1].imshow(cv2.cvtColor(vis1, cv2.COLOR_BGR2RGB))
        
        # Draw Epipolar line if F is provided
        if F is not None:
            p0 = np.array([qx, qy, 1.0])
            l_prime = F @ p0
            a, b, c = l_prime
            
            x0, x1 = 0, W_IMG
            if b != 0:
                y0 = int(-(a * x0 + c) / b)
                y1 = int(-(a * x1 + c) / b)
                axes[q_idx, 1].plot([x0, x1], [y0, y1], 'w--', linewidth=2, alpha=0.8)
            elif a != 0:
                x = int(-c / a)
                axes[q_idx, 1].plot([x, x], [0, H_IMG], 'w--', linewidth=2, alpha=0.8)

        title_parts = [f"#{r+1}: conf={c:.5f}" for r, c in enumerate(zip(top_confs))]
        axes[q_idx, 1].set_title(f'Image 1 — Top-{top_k} matches', fontsize=12)
        axes[q_idx, 1].axis('off')

    fig.suptitle(f'{pair_label}  |  conf_max={conf_matrix.max():.5f}', fontsize=14, y=1.01)
    plt.tight_layout()
    return fig


def main():
    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = 'mps'

    print(f"Device: {device}")
    print(f"Checkpoint: {CKPT_PATH}")
    print(f"Data: {DATA_DIR}")

    model = build_model(CKPT_PATH, device)

    os.makedirs('visualizations', exist_ok=True)

    for pair_idx in PAIR_INDICES:
        print(f"\nPair {pair_idx} (frames {pair_idx} & {pair_idx + FRAME_GAP})...")
        img0, img1, gray0, gray1, F_mat = load_pair(DATA_DIR, pair_idx, FRAME_GAP)

        t0 = torch.from_numpy(gray0).unsqueeze(0).unsqueeze(0).to(device)
        t1 = torch.from_numpy(gray1).unsqueeze(0).unsqueeze(0).to(device)

        data = {'image0': t0, 'image1': t1}
        with torch.no_grad():
            model.matcher(data)

        conf = data['conf_matrix'][0].cpu().numpy()  # [L, S]
        print(f"  conf_max={conf.max():.6f}  conf_min={conf.min():.8f}")

        fig = visualize_pair(img0, img1, conf, QUERY_POINTS, TOP_K,
                             f"Pair {pair_idx}", F=F_mat)

        out_path = f"visualizations/top{TOP_K}_pair{pair_idx}.png"
        fig.savefig(out_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved: {out_path}")

    print("\nDone! Check visualizations/ folder.")


if __name__ == '__main__':
    main()
