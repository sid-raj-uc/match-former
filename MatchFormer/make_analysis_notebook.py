import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []

# Cell 1: Imports and setup
cells.append(nbf.v4.new_markdown_cell("# MatchFormer Epipolar & Attention Analysis\nThis notebook evaluates multiple query points across various image pairs to visualize attention heatmaps (Layer 4), Ground Truth Epipolar geometry, and MatchFormer's predicted matching point."))
cells.append(nbf.v4.new_code_cell("""\
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
import glob
import os
import random

from config.defaultmf import get_cfg_defaults
from model.lightning_loftr import PL_LoFTR
from match_poses import read_trajectory, get_pose_for_image
from gt_epipolar import compute_fundamental_matrix, K
from compute_ear import compute_EAR, random_baseline_EAR

print("Imports successful.")
"""))

# Cell 2: Model Initialization
cells.append(nbf.v4.new_markdown_cell("## Model & Hook Initialization"))
cells.append(nbf.v4.new_code_cell("""\
config = get_cfg_defaults()
config.MATCHFORMER.BACKBONE_TYPE = 'litela'
config.MATCHFORMER.SCENS = 'indoor'
config.MATCHFORMER.RESOLUTION = (8, 4)
config.MATCHFORMER.COARSE.D_MODEL = 192
config.MATCHFORMER.COARSE.D_FFN = 192

model = PL_LoFTR(config, pretrained_ckpt='model/weights/indoor-lite-LA.ckpt')
model.eval()
print("Model loaded.")

cross_attn_matrices = {}
HW_layer_map = {'stage4_cross': (15, 20)}

def get_cross_attn(name, layer, ux, uy):
    def hook(model, input, output):
        x = input[0]
        B, N, C = x.shape
        MiniB = B // 2
        query = layer.q(x).reshape(B, N, layer.num_heads, C // layer.num_heads).permute(0, 1, 2, 3)
        kv = layer.kv(x).reshape(B, -1, 2, layer.num_heads, C // layer.num_heads).permute(2, 0, 1, 3, 4)
        if layer.cross:
            k1, k2 = kv[0].split(MiniB)
            key = torch.cat([k2, k1], dim=0) 
        else:
            return 
        Q = layer.feature_map(query).permute(0, 2, 1, 3) 
        K = layer.feature_map(key).permute(0, 2, 1, 3)
        H_feat, W_feat = HW_layer_map[name]
        feat_x = int((ux / 640.0) * W_feat)
        feat_y = int((uy / 480.0) * H_feat)
        query_idx = feat_y * W_feat + feat_x
        # Only compute dot product for the single query point to prevent OOM
        Q_single = Q[0, :, query_idx:query_idx+1, :] 
        K_targets = K[0, :, :, :]
        attn = torch.matmul(Q_single, K_targets.transpose(-2, -1))
        Z = 1 / (torch.einsum("hd,hd->h", Q_single[:, 0, :], K_targets.sum(dim=1)) + layer.eps)
        attn = attn * Z.view(-1, 1, 1) 
        cross_attn_matrices[name] = attn.mean(dim=0).squeeze(0).detach().cpu()
    return hook
"""))

# Cell 3: Image Pair Loader
cells.append(nbf.v4.new_markdown_cell("## Evaluate specific image pairs and points"))
cells.append(nbf.v4.new_code_cell("""\
def get_image(path, resize=(640, 480)):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    img_rgb = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
    img_rgb = cv2.resize(img_rgb, resize)
    img = cv2.resize(img, resize)
    img_tensor = torch.from_numpy(img).float() / 255.0
    img_tensor = img_tensor.unsqueeze(0).unsqueeze(0)
    return img_rgb, img_tensor

gt_file = '../tum_rgb_dataset/groundtruth.txt'
poses = read_trajectory(gt_file)
rgb_dir = '../tum_rgb_dataset/rgb/*.png'
all_imgs = sorted(glob.glob(rgb_dir))

def analyze_pair(img1_idx, img2_idx, query_points):
    img1_path = all_imgs[img1_idx]
    img2_path = all_imgs[img2_idx]
    
    img1_rgb, img1_tensor = get_image(img1_path)
    img2_rgb, img2_tensor = get_image(img2_path)
    
    T1 = get_pose_for_image(img1_path, poses)
    T2 = get_pose_for_image(img2_path, poses)
    
    if T1 is None or T2 is None:
        print("Missing GT poses for these timestamps.")
        return
        
    F = compute_fundamental_matrix(T1, T2, K, K)
    
    # We will compute the actual predicted match geometry from Loftr output too
    # MatchFormer inference
    input_data = {'image0': img1_tensor, 'image1': img2_tensor}
    
    for (ux, uy) in query_points:
        cross_attn_matrices.clear()
        hooks = [
            model.matcher.backbone.AttentionBlock4.block[2].attn.register_forward_hook(
                get_cross_attn('stage4_cross', model.matcher.backbone.AttentionBlock4.block[2].attn, ux, uy)
            )
        ]
        
        with torch.no_grad():
            model(input_data)
            
        for h in hooks: h.remove()
        
        # Get Predicted Match (Closest predicted query match)
        mkpts0 = input_data['mkpts0_f'].cpu().numpy()
        mkpts1 = input_data['mkpts1_f'].cpu().numpy()
        
        # Find closest match point
        query_pt = np.array([ux, uy])
        if len(mkpts0) > 0:
            dists = np.linalg.norm(mkpts0 - query_pt, axis=1)
            best_idx = np.argmin(dists)
            pred_match = mkpts1[best_idx]
            match_dist = dists[best_idx]
        else:
            pred_match = None
            match_dist = float('inf')
        
        # Setup Epipolar Math
        p = np.array([ux, uy, 1.0])
        l_prime = F @ p 
        a, b, c = l_prime
        
        attn_map = cross_attn_matrices['stage4_cross']
        H_feat, W_feat = HW_layer_map['stage4_cross']
        attn_heatmap = attn_map.reshape(H_feat, W_feat).numpy()
        attn_resized = cv2.resize(attn_heatmap, (640, 480))
        
        # Plotting
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        axes[0].imshow(img1_rgb)
        axes[0].plot(ux, uy, 'r*', markersize=15, label='Query')
        axes[0].legend()
        axes[0].set_title(f"Source Image (Pt: {ux}, {uy})")
        axes[0].axis('off')
        
        attn_norm = (attn_resized - attn_resized.min()) / (attn_resized.max() - attn_resized.min() + 1e-8)
        axes[1].imshow(img2_rgb)
        axes[1].imshow(attn_norm, cmap='jet', alpha=0.5)
        
        # Draw Epipolar line
        x0, x1 = 0, 640
        y0 = int(-(a*x0 + c) / b)
        y1 = int(-(a*x1 + c) / b)
        axes[1].plot([x0, x1], [y0, y1], 'w--', linewidth=2, label='Epipolar Line')
        
        # Draw Predicted Match
        if pred_match is not None and match_dist < 20.0:  # Only if a keypoint matched nearby our query
             axes[1].plot(pred_match[0], pred_match[1], 'g*', markersize=15, label='Predicted Match')
             
        axes[1].set_xlim([0, 640])
        axes[1].set_ylim([480, 0])
        axes[1].axis('off')
        
        # Calculate EAR metric comparison
        attn_prob = attn_resized / np.sum(attn_resized)
        ear_val = compute_EAR(attn_prob, a, b, c, 10)
        axes[1].set_title(f"Target: Stage 4 | EAR: {ear_val:.4f}")
        axes[1].legend(loc='lower right')
        
        plt.tight_layout()
        plt.show()

"""))

cells.append(nbf.v4.new_markdown_cell("## Run Tests on First Pair\nImage 0 -> Image 10"))
cells.append(nbf.v4.new_code_cell("""\
points = [
    (320, 240), # Center 
    (550, 100), # Blank Wall
    (300, 360)  # Keyboard
]
analyze_pair(0, 10, points)
"""))

cells.append(nbf.v4.new_markdown_cell("## Run Tests on Second Pair\nImage 20 -> Image 30"))
cells.append(nbf.v4.new_code_cell("""\
# Pick custom points across different textures based on what TUM sequence 1 looks like
points_2 = [
    (150, 150),
    (400, 300),
    (100, 400)
]
analyze_pair(20, 30, points_2)
"""))

nb.cells.extend(cells)

with open('Failure_Mode_Analysis.ipynb', 'w') as f:
    nbf.write(nb, f)

print("Created Failure_Mode_Analysis.ipynb successfully.")
