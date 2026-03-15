import nbformat as nbf
nb = nbf.v4.new_notebook()
cells = []

cells.append(nbf.v4.new_markdown_cell("# Point-by-Point Benchmark Visual Analysis\nComparing attention heatmaps and match precision for individual query points between Vanilla MatchFormer and Epipolar-Constrained MatchFormer."))
cells.append(nbf.v4.new_code_cell("""\
%matplotlib inline
import torch
import torch.nn as nn
import cv2
import numpy as np
import matplotlib.pyplot as plt
import glob

from config.defaultmf import get_cfg_defaults
from model.lightning_loftr import PL_LoFTR
from gt_epipolar import compute_fundamental_matrix
import os

# --- EPIPOLAR MASK ---
def create_epipolar_mask_flat(F, H_feat, W_feat, query_pt, H_img=480, W_img=640, tau=10.0):
    p = np.array([query_pt[0], query_pt[1], 1.0])
    l_prime = F @ p 
    a, b, c = l_prime
    y_feat, x_feat = np.mgrid[0:H_feat, 0:W_feat]
    x_img = (x_feat / W_feat) * W_img
    y_img = (y_feat / H_feat) * H_img
    distances = np.abs(a*x_img + b*y_img + c) / np.sqrt(a**2 + b**2)
    mask = np.exp(-distances / tau)
    return torch.from_numpy(mask.flatten()).float()

config = get_cfg_defaults()
config.MATCHFORMER.BACKBONE_TYPE = 'litela'
config.MATCHFORMER.SCENS = 'indoor'
config.MATCHFORMER.RESOLUTION = (8, 4)
config.MATCHFORMER.COARSE.D_MODEL = 192
config.MATCHFORMER.COARSE.D_FFN = 192

model_vanilla = PL_LoFTR(config, pretrained_ckpt='model/weights/indoor-lite-LA.ckpt').eval()
model_const = PL_LoFTR(config, pretrained_ckpt='model/weights/indoor-lite-LA.ckpt').eval()

HW_layer_map = {'stage4_cross': (15, 20)}

def get_cross_attn_hook(name, layer, ux, uy, store_dict, F_matrix=None):
    def hook(model, input, output):
        x = input[0]
        B, N, C = x.shape
        MiniB = B // 2
        query = layer.q(x).reshape(B, N, layer.num_heads, C // layer.num_heads).permute(0, 1, 2, 3)
        kv = layer.kv(x).reshape(B, -1, 2, layer.num_heads, C // layer.num_heads).permute(2, 0, 1, 3, 4)
        if layer.cross:
            k1, k2 = kv[0].split(MiniB)
            key = torch.cat([k2, k1], dim=0) 
        else: return 
        
        Q = layer.feature_map(query).permute(0, 2, 1, 3) 
        K = layer.feature_map(key).permute(0, 2, 1, 3)
        H_feat, W_feat = HW_layer_map[name]
        feat_x = int((ux / 640.0) * W_feat)
        feat_y = int((uy / 480.0) * H_feat)
        query_idx = feat_y * W_feat + feat_x

        Q_single = Q[0, :, query_idx:query_idx+1, :] 
        K_targets = K[0, :, :, :]
        attn = torch.matmul(Q_single, K_targets.transpose(-2, -1)) # (num_heads, 1, N)
        
        if F_matrix is not None:
             epipolar_mask = create_epipolar_mask_flat(F_matrix, H_feat, W_feat, (ux, uy))
             epipolar_mask = epipolar_mask.view(1, 1, -1).to(attn.device)
             attn = attn * epipolar_mask
             
        Z = 1 / (attn.sum(dim=-1, keepdim=True) + layer.eps)
        attn_final = attn * Z
        store_dict[name] = attn_final.mean(dim=0).squeeze(0).detach().cpu()
    return hook

data_dir = '../data/scans/scene0000_00/exported'
img_dir = f'{data_dir}/color/*.jpg'
all_imgs = sorted(glob.glob(img_dir))
# Use intrinsic_depth because it corresponds to the exactly 640x480 resolution we resize to.
K = np.loadtxt(f'{data_dir}/intrinsic/intrinsic_depth.txt')[:3, :3]

def get_image(path, resize=(640, 480)):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    img_rgb = cv2.resize(cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB), resize)
    img_rgb = cv2.resize(img_rgb, resize)
    img = cv2.resize(img, resize)
    img_tensor = torch.from_numpy(img).float() / 255.0
    img_tensor = img_tensor.unsqueeze(0).unsqueeze(0)
    return img_rgb, img_tensor

def compute_ear(attn_prob, a, b, c, k_pixels):
    H, W = attn_prob.shape
    y, x = np.mgrid[0:H, 0:W]
    distances = np.abs(a*x + b*y + c) / np.sqrt(a**2 + b**2)
    mask = distances <= k_pixels
    return np.sum(attn_prob[mask])

def get_closest_match(mkpts0, mkpts1, query_pt):
    if len(mkpts0) == 0: return None
    dists = np.linalg.norm(mkpts0 - query_pt, axis=1)
    best_idx = np.argmin(dists)
    if dists[best_idx] < 20.0:  # Must have a keypoint nearby to count
         return mkpts1[best_idx]
    return None

def draw_epipolar_line(ax, a, b, c, w=640, h=480):
    x0, x1 = 0, w
    y0 = int(-(a*x0 + c) / b)
    y1 = int(-(a*x1 + c) / b)
    ax.plot([x0, x1], [y0, y1], 'w--', linewidth=2, label='Epipolar Line')

def get_gt_match(pt1, depth1_path, pose1, pose2, K):
    # Load depth image
    depth1 = cv2.imread(depth1_path, cv2.IMREAD_ANYDEPTH)
    if depth1 is None: return None
    
    # Resize depth to 640x480 if not already
    if depth1.shape != (480, 640):
        depth1 = cv2.resize(depth1, (640, 480), interpolation=cv2.INTER_NEAREST)
        
    z = depth1[int(pt1[1]), int(pt1[0])] / 1000.0 # Convert mm to meters
    if z <= 0.1 or z > 10.0: return None # Invalid depth
    
    # 1. Pixel to Camera 1 coordinates
    cx, cy = K[0, 2], K[1, 2]
    fx, fy = K[0, 0], K[1, 1]
    x_c1 = (pt1[0] - cx) * z / fx
    y_c1 = (pt1[1] - cy) * z / fy
    p_c1 = np.array([x_c1, y_c1, z, 1.0])
    
    # 2. Transform Camera 1 -> World -> Camera 2
    # Poses from ScanNet are Camera-to-World (T_WC) in OpenGL coordinates.
    # We must convert to OpenCV coordinates (+Z forward)
    T_cv2gl = np.array([[1,0,0,0],[0,-1,0,0],[0,0,-1,0],[0,0,0,1]])
    T_12 = T_cv2gl @ np.linalg.inv(pose2) @ pose1 @ T_cv2gl
    p_c2 = T_12 @ p_c1
    
    # 3. Camera 2 to Pixel coordinates
    if p_c2[2] <= 0: return None # Behind camera 2
    u2 = (p_c2[0] * fx / p_c2[2]) + cx
    v2 = (p_c2[1] * fy / p_c2[2]) + cy
    
    # Check bounds
    if 0 <= u2 < 640 and 0 <= v2 < 480:
        return np.array([u2, v2])
    return None

def plot_triplet(img1_idx, img2_idx, query_points):
    img1_path = all_imgs[img1_idx]
    img2_path = all_imgs[img2_idx]
    img1_rgb, img1_tensor = get_image(img1_path)
    img2_rgb, img2_tensor = get_image(img2_path)
    
    img1_idx_num = os.path.basename(img1_path).split('.')[0]
    img2_idx_num = os.path.basename(img2_path).split('.')[0]
    try:
        T1 = np.loadtxt(os.path.join(data_dir, 'pose', f'{img1_idx_num}.txt'))
        T2 = np.loadtxt(os.path.join(data_dir, 'pose', f'{img2_idx_num}.txt'))
    except FileNotFoundError:
        return
    
    # Check if pose contains inf or nan
    if not np.isfinite(T1).all() or not np.isfinite(T2).all(): return
        
    T_cv2gl = np.array([[1,0,0,0],[0,-1,0,0],[0,0,-1,0],[0,0,0,1]])
    F = compute_fundamental_matrix(T1 @ T_cv2gl, T2 @ T_cv2gl, K, K)
    
    depth1_path = os.path.join(data_dir, 'depth', f'{img1_idx_num}.png')
    
    input_data_v = {'image0': img1_tensor, 'image1': img2_tensor}
    input_data_c = {'image0': img1_tensor, 'image1': img2_tensor}
    
    for (pt_name, (ux, uy)) in query_points.items():
        vanilla_attn = {}
        const_attn = {}
        
        # Hooks
        h_v = model_vanilla.matcher.backbone.AttentionBlock4.block[2].attn.register_forward_hook(
            get_cross_attn_hook('stage4_cross', model_vanilla.matcher.backbone.AttentionBlock4.block[2].attn, ux, uy, vanilla_attn, None)
        )
        h_c = model_const.matcher.backbone.AttentionBlock4.block[2].attn.register_forward_hook(
            get_cross_attn_hook('stage4_cross', model_const.matcher.backbone.AttentionBlock4.block[2].attn, ux, uy, const_attn, F)
        )
        
        with torch.no_grad():
            model_vanilla.matcher(input_data_v)
            model_const.matcher.coarse_matching.epipolar_F = F
            model_const.matcher(input_data_c)
            
        h_v.remove()
        h_c.remove()
        
        # Skip mkpts altogether and use heatmap peaks directly
        
        # Math Epipolar
        p = np.array([ux, uy, 1.0])
        l_prime = F @ p 
        a, b, c = l_prime
        
        H_feat, W_feat = HW_layer_map['stage4_cross']
        
        heat_v = vanilla_attn['stage4_cross'].reshape(H_feat, W_feat).numpy()
        heat_v_rez = cv2.resize(heat_v, (640, 480))
        heat_v_norm = (heat_v_rez - heat_v_rez.min()) / (heat_v_rez.max() - heat_v_rez.min() + 1e-8)
        ear_v = compute_ear(heat_v_rez / np.sum(heat_v_rez), a, b, c, 10)
        max_idx_v = np.unravel_index(np.argmax(heat_v_rez), heat_v_rez.shape)
        pred_v = np.array([max_idx_v[1], max_idx_v[0]])
        
        heat_c = const_attn['stage4_cross'].reshape(H_feat, W_feat).numpy()
        heat_c_rez = cv2.resize(heat_c, (640, 480))
        heat_c_norm = (heat_c_rez - heat_c_rez.min()) / (heat_c_rez.max() - heat_c_rez.min() + 1e-8)
        ear_c = compute_ear(heat_c_rez / np.sum(heat_c_rez), a, b, c, 10)
        max_idx_c = np.unravel_index(np.argmax(heat_c_rez), heat_c_rez.shape)
        pred_c = np.array([max_idx_c[1], max_idx_c[0]])
        
        # Compute GT Match
        gt_match = get_gt_match((ux, uy), depth1_path, T1, T2, K)
        
        # ----- PLOTTING -----
        fig, axes = plt.subplots(1, 3, figsize=(20, 6))
        
        # 1. Source
        axes[0].imshow(img1_rgb)
        axes[0].plot(ux, uy, 'r*', markersize=15)
        axes[0].set_title(f"Source Image\\nQuery: {pt_name}")
        axes[0].axis('off')
        
        # 2. Vanilla
        axes[1].imshow(img2_rgb)
        axes[1].imshow(heat_v_norm, cmap='jet', alpha=0.5)
        draw_epipolar_line(axes[1], a, b, c)
        if gt_match is not None:
             axes[1].plot(gt_match[0], gt_match[1], 'yo', markersize=12, label='Ground Truth')
        if pred_v is not None:
             if gt_match is not None:
                 dist_v = np.linalg.norm(pred_v - gt_match)
                 color = 'g*' if dist_v < 10 else 'rx'
                 axes[1].plot(pred_v[0], pred_v[1], color, markersize=12, label=f'Pred (GT Err: {dist_v:.1f}px)')
             else:
                 dist_v = np.abs(a*pred_v[0] + b*pred_v[1] + c) / np.sqrt(a**2 + b**2)
                 color = 'g*' if dist_v < 5.0 else 'rx'
                 axes[1].plot(pred_v[0], pred_v[1], color, markersize=12, label=f'Pred (Epi Err: {dist_v:.1f}px)')
        axes[1].set_title(f"Vanilla Target\\nEAR: {ear_v:.4f}")
        axes[1].legend(loc='lower right')
        axes[1].set_xlim([0, 640]); axes[1].set_ylim([480, 0])
        axes[1].axis('off')
        
        # 3. Constrained
        axes[2].imshow(img2_rgb)
        axes[2].imshow(heat_c_norm, cmap='jet', alpha=0.5)
        draw_epipolar_line(axes[2], a, b, c)
        if gt_match is not None:
             axes[2].plot(gt_match[0], gt_match[1], 'yo', markersize=12, label='Ground Truth')
        if pred_c is not None:
             if gt_match is not None:
                 dist_c = np.linalg.norm(pred_c - gt_match)
                 color = 'g*' if dist_c < 10 else 'rx'
                 axes[2].plot(pred_c[0], pred_c[1], color, markersize=12, label=f'Pred (GT Err: {dist_c:.1f}px)')
             else:
                 dist_c = np.abs(a*pred_c[0] + b*pred_c[1] + c) / np.sqrt(a**2 + b**2)
                 color = 'g*' if dist_c < 5.0 else 'rx'
                 axes[2].plot(pred_c[0], pred_c[1], color, markersize=12, label=f'Pred (Epi Err: {dist_c:.1f}px)')
        axes[2].set_title(f"Our Model Target\\nEAR: {ear_c:.4f}")
        axes[2].legend(loc='lower right')
        axes[2].set_xlim([0, 640]); axes[2].set_ylim([480, 0])
        axes[2].axis('off')
        
        plt.tight_layout()
        plt.show()

print("Setup Complete.")
"""))

cells.append(nbf.v4.new_markdown_cell("## Points Test"))
cells.append(nbf.v4.new_code_cell("""\
np.random.seed(42)  # For reproducibility

print("Running Triplet Analysis for 20 distinct points across random image pairs...")
valid_pairs = 0
attempts = 0

while valid_pairs < 20 and attempts < 100:
    attempts += 1
    idx1 = np.random.randint(0, len(all_imgs) - 20)
    idx2 = idx1 + 20  # Keep a 20 frame baseline gap
    
    # Check if poses exist for both (by trying to load)
    file_idx1 = os.path.basename(all_imgs[idx1]).split('.')[0]
    file_idx2 = os.path.basename(all_imgs[idx2]).split('.')[0]
    if not os.path.exists(os.path.join(data_dir, 'pose', f'{file_idx1}.txt')) or \\
       not os.path.exists(os.path.join(data_dir, 'pose', f'{file_idx2}.txt')):
        continue
        
    x = np.random.randint(50, 590)
    y = np.random.randint(50, 430)
    pt_dict = {f'Random_Pair_{valid_pairs}': (x, y)}
    
    plot_triplet(idx1, idx2, pt_dict)
    valid_pairs += 1
"""))

nb.cells.extend(cells)

with open('Triplet_Visual_Analysis.ipynb', 'w') as f:
    nbf.write(nb, f)

print("Created Triplet Notebook")
