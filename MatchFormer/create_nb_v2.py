import nbformat as nbf

nb = nbf.v4.new_notebook()

nb.cells = [
    nbf.v4.new_markdown_cell("# WxBS Visual Analysis with MatchFormer\nThis notebook computes the Fundamental matrix from ground truth correspondences, plots the epipolar line, and compares the GT match with the MatchFormer (outdoor-large-LA) prediction."),
    nbf.v4.new_code_cell("""\
import os
import sys
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt

# Add MatchFormer to path so we can import its modules
sys.path.insert(0, os.path.abspath('.'))
from config.defaultmf import get_cfg_defaults
from model.lightning_loftr import PL_LoFTR

DATA_DIR = '/Users/siddharthraj/classes/cv/cv_final/data/WxBS_data_folder/v1.1'
CKPT_PATH = '/Users/siddharthraj/classes/cv/cv_final/MatchFormer/model/weights/outdoor-large-LA.ckpt'

H_IMG, W_IMG = 480, 640
STRIDE = 8
H_C, W_C = H_IMG // STRIDE, W_IMG // STRIDE
"""),
    nbf.v4.new_code_cell("""\
def build_model(ckpt_path, device='cpu'):
    config = get_cfg_defaults()
    config.MATCHFORMER.BACKBONE_TYPE = 'largela'
    config.MATCHFORMER.SCENS = 'outdoor'
    config.MATCHFORMER.RESOLUTION = (8, 2)
    config.MATCHFORMER.COARSE.D_MODEL = 256
    config.MATCHFORMER.COARSE.D_FFN = 256
    
    model = PL_LoFTR(config, pretrained_ckpt=ckpt_path)
    model.to(device).eval()
    return model

device = 'cuda' if torch.cuda.is_available() else 'mps' if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available() else 'cpu'
print(f"Loading model on {device}...")
model = build_model(CKPT_PATH, device)
print("Model loaded.")
"""),
    nbf.v4.new_code_cell("""\
def load_and_resize_pair(category, pair_name):
    base_path = os.path.join(DATA_DIR, category, pair_name)
    img1_path = os.path.join(base_path, '01.png')
    img2_path = os.path.join(base_path, '02.png')
    corrs_path = os.path.join(base_path, 'corrs.txt')
    
    img1_raw = cv2.imread(img1_path)
    img2_raw = cv2.imread(img2_path)
    
    h1, w1 = img1_raw.shape[:2]
    h2, w2 = img2_raw.shape[:2]
    
    img1 = cv2.cvtColor(cv2.resize(img1_raw, (W_IMG, H_IMG)), cv2.COLOR_BGR2RGB)
    img2 = cv2.cvtColor(cv2.resize(img2_raw, (W_IMG, H_IMG)), cv2.COLOR_BGR2RGB)
    
    gray1 = cv2.cvtColor(cv2.resize(img1_raw, (W_IMG, H_IMG)), cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    gray2 = cv2.cvtColor(cv2.resize(img2_raw, (W_IMG, H_IMG)), cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    
    corrs = np.loadtxt(corrs_path)
    pts1 = corrs[:, :2]
    pts2 = corrs[:, 2:]
    
    # Scale points to new dimensions
    pts1_scaled = pts1 * np.array([W_IMG / w1, H_IMG / h1])
    pts2_scaled = pts2 * np.array([W_IMG / w2, H_IMG / h2])
    
    return img1, img2, gray1, gray2, pts1_scaled, pts2_scaled
"""),
    nbf.v4.new_code_cell("""\
def pixel_to_coarse(px, py):
    cx = int(px / STRIDE)
    cy = int(py / STRIDE)
    cx = min(cx, W_C - 1)
    cy = min(cy, H_C - 1)
    return cy * W_C + cx

def coarse_to_pixel(linear_idx):
    cx = linear_idx % W_C
    cy = linear_idx // W_C
    px = cx * STRIDE + STRIDE // 2
    py = cy * STRIDE + STRIDE // 2
    return px, py

def get_model_prediction(gray1, gray2, p1):
    # Run inference
    t0 = torch.from_numpy(gray1).unsqueeze(0).unsqueeze(0).to(device)
    t1 = torch.from_numpy(gray2).unsqueeze(0).unsqueeze(0).to(device)
    data = {'image0': t0, 'image1': t1}
    with torch.no_grad():
        model.matcher(data)
    
    conf = data['conf_matrix'][0].cpu().numpy()
    
    # Get coarse index for query point p1 = (x, y)
    qx, qy = p1
    q_linear = pixel_to_coarse(qx, qy)
    row = conf[q_linear, :]
    
    # Find the best match
    best_idx = np.argmax(row)
    mx, my = coarse_to_pixel(best_idx)
    
    return (mx, my), row[best_idx]
"""),
    nbf.v4.new_code_cell("""\
def visualize_wxbs_analysis(category, pair_name, pt_idx=0):
    img1, img2, gray1, gray2, pts1, pts2 = load_and_resize_pair(category, pair_name)
    
    # Compute F
    F, _ = cv2.findFundamentalMat(pts1, pts2, cv2.FM_8POINT)
    
    # Pick the GT point
    x1, y1 = pts1[pt_idx]
    x2, y2 = pts2[pt_idx]
    
    # Get model prediction
    (pred_x, pred_y), conf = get_model_prediction(gray1, gray2, (x1, y1))
    
    # Epipolar line in img2: l' = F * p1
    p1 = np.array([x1, y1, 1.0])
    l_prime = F @ p1
    a, b, c = l_prime
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Image 1
    axes[0].imshow(img1)
    axes[0].plot(x1, y1, 'r*', markersize=12)
    axes[0].set_title(f"1. Query Point\\n{category}/{pair_name}")
    axes[0].axis('off')
    
    # Epipolar line drawing helper
    def draw_epipolar(ax, color='w--'):
        h, w = img2.shape[:2]
        x0, x_end = 0, w
        if b != 0:
            y0 = int(-(a * x0 + c) / b)
            y_end = int(-(a * x_end + c) / b)
            ax.plot([x0, x_end], [y0, y_end], color, linewidth=1, alpha=0.8, label='Epipolar Line')
        elif a != 0:
            x_val = int(-c / a)
            ax.plot([x_val, x_val], [0, h], color, linewidth=1, alpha=0.8, label='Epipolar Line')

    # Image 2 (GT Match)
    axes[1].imshow(img2)
    draw_epipolar(axes[1])
    axes[1].plot(x2, y2, 'g*', markersize=12, label='GT Match')
    axes[1].set_title("2. Truth & Epipolar Line")
    axes[1].legend()
    axes[1].axis('off')
    
    # Image 3 (Prediction)
    axes[2].imshow(img2)
    draw_epipolar(axes[2])
    axes[2].plot(pred_x, pred_y, 'b*', markersize=12, label=f'Predicted (conf={conf:.3f})')
    # Draw GT as well as reference
    axes[2].plot(x2, y2, 'g.', markersize=8, alpha=0.5, label='GT')
    axes[2].set_title("3. MatchFormer Pred & Epipolar Line")
    axes[2].legend()
    axes[2].axis('off')

    plt.tight_layout()
    plt.show()
"""),
    nbf.v4.new_code_cell("""\
# Analyze a randomly selected point from WGALBS/bridge
visualize_wxbs_analysis('WGALBS', 'bridge', pt_idx=10)
""")
]

with open('WxBS_Visual_Analysis.ipynb', 'w') as f:
    nbf.write(nb, f)
print("Updated Notebook successfully created!")
