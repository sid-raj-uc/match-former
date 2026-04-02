import os
import sys
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath('.'))
from config.defaultmf import get_cfg_defaults
from model.lightning_loftr import PL_LoFTR

DATA_DIR = '/Users/siddharthraj/classes/cv/cv_final/data/WxBS_data_folder/v1.1'
CKPT_PATH = '/Users/siddharthraj/classes/cv/cv_final/MatchFormer/model/weights/outdoor-large-LA.ckpt'

H_IMG, W_IMG = 480, 640
STRIDE = 8
H_C, W_C = H_IMG // STRIDE, W_IMG // STRIDE

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
    if len(corrs) == 0:
        return None
    pts1 = corrs[:, :2]
    pts2 = corrs[:, 2:]
    
    # Scale points to new dimensions
    pts1_scaled = pts1 * np.array([W_IMG / w1, H_IMG / h1])
    pts2_scaled = pts2 * np.array([W_IMG / w2, H_IMG / h2])
    
    return img1, img2, gray1, gray2, pts1_scaled, pts2_scaled

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

def visualize_wxbs_analysis(category, pair_name, pt_idx=0):
    res = load_and_resize_pair(category, pair_name)
    if res is None:
        print(f"Warning: No correspondences found for {category}/{pair_name}")
        return
    img1, img2, gray1, gray2, pts1, pts2 = res
    
    # Compute F
    F, _ = cv2.findFundamentalMat(pts1, pts2, cv2.FM_8POINT)
    print("F Computed")
    
    # Pick the GT point
    x1, y1 = pts1[pt_idx]
    x2, y2 = pts2[pt_idx]
    
    # Get model prediction
    (pred_x, pred_y), conf = get_model_prediction(gray1, gray2, (x1, y1))
    print("Prediction got")
    
    # Calculate pixel distance error
    error_px = np.linalg.norm(np.array([pred_x, pred_y]) - np.array([x2, y2]))
    
    # Epipolar line in img2: l' = F * p1
    p1 = np.array([x1, y1, 1.0])
    l_prime = F @ p1
    a, b, c = l_prime
    print("Complete")

# Analyze multiple random scenes across categories
CATEGORIES = ['WGALBS']
for i in range(1):
    cat = 'WGALBS'
    pair = 'bridge'
    corrs = np.loadtxt(os.path.join(DATA_DIR, cat, pair, 'corrs.txt'))
    pt_idx = 0
    print(f"\n--- [Pair {i+1}] Analyzing {cat}/{pair} | Point {pt_idx} ---")
    visualize_wxbs_analysis(cat, pair, pt_idx)
