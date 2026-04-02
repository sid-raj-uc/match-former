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
    
    def find_img(prefix):
        for ext in ['.png', '.jpg', '.jpeg', '.ppm']:
            p = os.path.join(base_path, prefix + ext)
            if os.path.exists(p): return p
        return None
        
    img1_path = find_img('01')
    img2_path = find_img('02')
    corrs_path = os.path.join(base_path, 'corrs.txt')
    
    if not img1_path or not img2_path:
        return None
        
    img1_raw = cv2.imread(img1_path)
    img2_raw = cv2.imread(img2_path)
    
    h1, w1 = img1_raw.shape[:2]
    h2, w2 = img2_raw.shape[:2]
    
    img1 = cv2.cvtColor(cv2.resize(img1_raw, (W_IMG, H_IMG)), cv2.COLOR_BGR2RGB)
    img2 = cv2.cvtColor(cv2.resize(img2_raw, (W_IMG, H_IMG)), cv2.COLOR_BGR2RGB)
    
    gray1 = cv2.cvtColor(cv2.resize(img1_raw, (W_IMG, H_IMG)), cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    gray2 = cv2.cvtColor(cv2.resize(img2_raw, (W_IMG, H_IMG)), cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    
    try:
        corrs = np.loadtxt(corrs_path)
        if len(corrs) == 0:
            return None
        # Handle case where there's only 1 correspondence (1D array)
        if corrs.ndim == 1:
            corrs = corrs[np.newaxis, :]
        pts1 = corrs[:, :2]
        pts2 = corrs[:, 2:]
    except Exception:
        return None
    
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
        print(f"Warning: No correspondences or images found for {category}/{pair_name}")
        return
    img1, img2, gray1, gray2, pts1, pts2 = res
    
    # Compute F
    F, _ = cv2.findFundamentalMat(pts1, pts2, cv2.FM_8POINT)
    
    if F is None or F.shape != (3, 3):
        print(f"Warning: Fundamental matrix estimation failed for {category}/{pair_name}")
        return
    
    # Pick the GT point
    x1, y1 = pts1[pt_idx]
    x2, y2 = pts2[pt_idx]
    
    # Get model prediction
    (pred_x, pred_y), conf = get_model_prediction(gray1, gray2, (x1, y1))
    
    # Calculate pixel distance error
    error_px = np.linalg.norm(np.array([pred_x, pred_y]) - np.array([x2, y2]))
    
    # Epipolar line in img2: l' = F * p1
    p1 = np.array([x1, y1, 1.0])
    l_prime = F @ p1
    a, b, c = l_prime
    
    fig, axes = plt.subplots(1, 3, figsize=(22, 7))
    
    # Image 1
    axes[0].imshow(img1)
    axes[0].plot(x1, y1, 'r*', markersize=12)
    axes[0].set_title(f"1. Query Point\n{category}/{pair_name}")
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
    axes[2].plot(pred_x, pred_y, 'b*', markersize=12, label='Predicted Match')
    # Draw GT as well as reference
    axes[2].plot(x2, y2, 'g.', markersize=8, alpha=0.5, label='GT Point Overlay')
    axes[2].set_title(f"3. MatchFormer Output\nErr: {error_px:.1f} px | Conf: {conf:.3f}")
    axes[2].legend()
    axes[2].axis('off')

    plt.tight_layout()
    plt.show()

# Analyze multiple random scenes across categories
CATEGORIES = ['WGALBS', 'WGBS', 'WLABS', 'WGSBS', 'WGABS']
np.random.seed(42)

for i in range(5):
    # Randomly select category
    cat = np.random.choice(CATEGORIES)
    cat_dir = os.path.join(DATA_DIR, cat)
    
    if not os.path.exists(cat_dir):
        continue
        
    # Randomly select a pair directory within the category
    pairs = [d for d in os.listdir(cat_dir) if os.path.isdir(os.path.join(cat_dir, d))]
    if not pairs:
        continue
        
    pair = np.random.choice(pairs)
    corrs_path = os.path.join(cat_dir, pair, 'corrs.txt')
    
    if not os.path.exists(corrs_path):
        continue
        
    try:
        corrs = np.loadtxt(corrs_path)
    except:
        continue
        
    # Handle single dimension arrays if there is only 1 point match
    if corrs.ndim == 1 and len(corrs) > 0:
        corrs = corrs[np.newaxis, :]
        
    if len(corrs) > 0:
        pt_idx = np.random.randint(len(corrs))
        print(f"\n--- [Pair {i+1}] Analyzing {cat}/{pair} | Point {pt_idx} ---")
        visualize_wxbs_analysis(cat, pair, pt_idx)

def get_epipolar_distance(F, p1, p2):
    # p1, p2 are (x, y)
    h1 = np.array([p1[0], p1[1], 1.0])
    h2 = np.array([p2[0], p2[1], 1.0])
    
    l2 = F @ h1
    d2 = np.abs(np.dot(h2, l2)) / np.sqrt(l2[0]**2 + l2[1]**2 + 1e-8)
    
    l1 = F.T @ h2
    d1 = np.abs(np.dot(h1, l1)) / np.sqrt(l1[0]**2 + l1[1]**2 + 1e-8)
    
    return (d1 + d2) / 2.0

def visualize_top_global_matches(category, pair_name, top_n=10):
    res = load_and_resize_pair(category, pair_name)
    if res is None:
        print(f"Skipping {category}/{pair_name}...")
        return
    img1, img2, gray1, gray2, pts1, pts2 = res

    # Compute Fundamental Matrix F
    F, _ = cv2.findFundamentalMat(pts1, pts2, cv2.FM_8POINT)
    if F is None or F.shape != (3, 3):
        print(f"Fundamental matrix estimation failed for {category}/{pair_name}")
        return

    t0 = torch.from_numpy(gray1).unsqueeze(0).unsqueeze(0).to(device)
    t1 = torch.from_numpy(gray2).unsqueeze(0).unsqueeze(0).to(device)
    data = {'image0': t0, 'image1': t1}
    with torch.no_grad():
        model.matcher(data)

    mkpts0 = data['mkpts0_f'].cpu().numpy()
    mkpts1 = data['mkpts1_f'].cpu().numpy()
    mconf = data['mconf'].cpu().numpy()

    # Sort by confidence
    sort_idx = np.argsort(mconf)[::-1]
    top_idx = sort_idx[:min(top_n, len(mconf))]

    top_pts0 = mkpts0[top_idx]
    top_pts1 = mkpts1[top_idx]
    top_confs = mconf[top_idx]

    # Calculate Epipolar Pixel Error for each match
    errors = []
    for i in range(len(top_pts0)):
        err = get_epipolar_distance(F, top_pts0[i], top_pts1[i])
        errors.append(err)
    
    mean_error = np.mean(errors) if len(errors) > 0 else 0

    # Create a concatenated image for side-by-side match drawing
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    vis_img = np.zeros((max(h1, h2), w1 + w2, 3), dtype=np.uint8)
    vis_img[:h1, :w1] = img1
    vis_img[:h2, w1:w1+w2] = img2

    fig, ax = plt.subplots(figsize=(18, 9))
    ax.imshow(vis_img)

    colors = plt.cm.jet(np.linspace(0, 1, top_n))
    for i in range(len(top_pts0)):
        x1, y1 = top_pts0[i]
        x2, y2 = top_pts1[i]
        c = colors[i]
        err = errors[i]
        ax.plot([x1, x2 + w1], [y1, y2], color=c, linewidth=2, marker='o', markersize=6,
                label=f'Rank {i+1} (conf: {top_confs[i]:.3f}) | Err: {err:.1f}px')

    ax.axis('off')
    ax.set_title(f"Top {top_n} Matches | {category}/{pair_name} | Mean Err: {mean_error:.2f}px", fontsize=18)
    
    # Draw legend outside the plot box
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
    plt.tight_layout()
    plt.show()


print("Visualizing Top 10 Matches for 3 random pairs:")
np.random.seed(1337) # different seed for variety
for _ in range(3):
    cat = np.random.choice(CATEGORIES)
    cat_dir = os.path.join(DATA_DIR, cat)
    if not os.path.exists(cat_dir): continue
    pairs = [d for d in os.listdir(cat_dir) if os.path.isdir(os.path.join(cat_dir, d))]
    if not pairs: continue
    pair = np.random.choice(pairs)
    
    visualize_top_global_matches(cat, pair, top_n=10)

def visualize_masked_vs_unmasked(category, pair_name, top_n=10):
    res = load_and_resize_pair(category, pair_name)
    if res is None:
        print(f"Skipping {category}/{pair_name}...")
        return
    img1, img2, gray1, gray2, pts1, pts2 = res

    F, _ = cv2.findFundamentalMat(pts1, pts2, cv2.FM_8POINT)
    if F is None or F.shape != (3, 3):
        return

    t0 = torch.from_numpy(gray1).unsqueeze(0).unsqueeze(0).to(device)
    t1 = torch.from_numpy(gray2).unsqueeze(0).unsqueeze(0).to(device)
    data = {'image0': t0, 'image1': t1}
    
    # 1. Vanilla (Unmasked) Execution
    model.matcher.coarse_matching.epipolar_F = None
    with torch.no_grad():
        model.matcher(data)

    mconf_raw = data['mconf'].cpu().numpy()
    sort_idx_raw = np.argsort(mconf_raw)[::-1][:min(top_n, len(mconf_raw))]
    pts0_raw = data['mkpts0_f'].cpu().numpy()[sort_idx_raw]
    pts1_raw = data['mkpts1_f'].cpu().numpy()[sort_idx_raw]
    
    # 2. Epipolar Masked Execution
    model.matcher.coarse_matching.epipolar_F = F
    with torch.no_grad():
        model.matcher(data)
        
    mconf_masked = data['mconf'].cpu().numpy()
    sort_idx_masked = np.argsort(mconf_masked)[::-1][:min(top_n, len(mconf_masked))]
    pts0_masked = data['mkpts0_f'].cpu().numpy()[sort_idx_masked]
    pts1_masked = data['mkpts1_f'].cpu().numpy()[sort_idx_masked]
    
    def draw_points_error(ax, p0_arr, p1_arr, title):
        h1, w1 = img1.shape[:2]
        h2, w2 = img2.shape[:2]
        vis_img = np.zeros((max(h1, h2), w1 + w2, 3), dtype=np.uint8)
        vis_img[:h1, :w1] = img1
        vis_img[:h2, w1:w1+w2] = img2
        ax.imshow(vis_img)
        
        errors = []
        colors = plt.cm.jet(np.linspace(0, 1, len(p0_arr)))
        for i in range(len(p0_arr)):
            x1, y1 = p0_arr[i]
            x2, y2 = p1_arr[i]
            c = colors[i]
            err = get_epipolar_distance(F, p0_arr[i], p1_arr[i])
            errors.append(err)
            ax.plot([x1, x2 + w1], [y1, y2], color=c, linewidth=2, marker='o', markersize=6)
            
        mean_err = np.mean(errors) if len(errors) > 0 else 0
        ax.set_title(f"{title}\nMean Err: {mean_err:.2f}px", fontsize=16)
        ax.axis('off')
        
    fig, axes = plt.subplots(2, 1, figsize=(18, 16))
    draw_points_error(axes[0], pts0_raw, pts1_raw, "Vanilla MatchFormer (Unmasked)")
    draw_points_error(axes[1], pts0_masked, pts1_masked, "Native Epipolar Enforced")
    plt.suptitle(f"Epipolar Overlap Analysis | {category}/{pair_name}", fontsize=22)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


print("Visualizing Masked vs Unmasked Top 10 Predictions:")


