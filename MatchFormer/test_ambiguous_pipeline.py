import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load our utilities
from config.defaultmf import get_cfg_defaults
from model.lightning_loftr import PL_LoFTR
from match_poses import read_trajectory, get_pose_for_image
from gt_epipolar import compute_fundamental_matrix, K
from compute_ear import compute_EAR, random_baseline_EAR

# 1. Initialize Model
config = get_cfg_defaults()
config.MATCHFORMER.BACKBONE_TYPE = 'litela'
config.MATCHFORMER.SCENS = 'indoor'
config.MATCHFORMER.RESOLUTION = (8, 4)
config.MATCHFORMER.COARSE.D_MODEL = 192
config.MATCHFORMER.COARSE.D_FFN = 192

model = PL_LoFTR(config, pretrained_ckpt='model/weights/indoor-lite-LA.ckpt')
model.eval()

# 2. Add Hooks - Modified to save memory by only computing a single query mapping instead of all points
cross_attn_matrices = {}

HW_layer_map = {
   'stage1_cross': (120, 160), # 480/4 = 120, 640/4 = 160
   'stage2_cross': (60, 80),   # /8
   'stage3_cross': (30, 40), # /16
   'stage4_cross': (15, 20)  # /32
}

def get_cross_attn(name, layer, ux, uy):
    def hook(model, input, output):
        x = input[0]
        B, N, C = x.shape
        MiniB = B // 2
        
        query = layer.q(x).reshape(B, N, layer.num_heads, C // layer.num_heads).permute(0, 1, 2, 3)
        kv = layer.kv(x).reshape(B, -1, 2, layer.num_heads, C // layer.num_heads).permute(2, 0, 1, 3, 4)
        
        if layer.cross == True:
            k1, k2 = kv[0].split(MiniB)
            key = torch.cat([k2, k1], dim=0) # (B, N, num_heads, head_dim)
        else:
            return 
            
        Q = layer.feature_map(query).permute(0, 2, 1, 3) # (B, num_heads, N, head_dim)
        K = layer.feature_map(key).permute(0, 2, 1, 3)
        
        H_feat, W_feat = HW_layer_map[name]
        feat_x = int((ux / 640.0) * W_feat)
        feat_y = int((uy / 480.0) * H_feat)
        query_idx = feat_y * W_feat + feat_x

        Q_single = Q[0, :, query_idx:query_idx+1, :] # Shape: (num_heads, 1, head_dim)
        K_targets = K[0, :, :, :] # Shape: (num_heads, N, head_dim)
        
        attn = torch.matmul(Q_single, K_targets.transpose(-2, -1))
        
        Z = 1 / (torch.einsum("hd,hd->h", Q_single[:, 0, :], K_targets.sum(dim=1)) + layer.eps)
        attn = attn * Z.view(-1, 1, 1) # -> (num_heads, 1, N)
        
        cross_attn_matrices[name] = attn.mean(dim=0).squeeze(0).detach().cpu()
        
    return hook

# 3. Load Images & GT Poses
img1_path = '../tum_rgb_dataset/rgb/1305031125.479424.png'
img2_path = '../tum_rgb_dataset/rgb/1305031113.411625.png'

def get_image(path, resize=(640, 480)):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    img_rgb = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
    img_rgb = cv2.resize(img_rgb, resize)
    img = cv2.resize(img, resize)
    img_tensor = torch.from_numpy(img).float() / 255.0
    img_tensor = img_tensor.unsqueeze(0).unsqueeze(0)
    return img_rgb, img_tensor

img1_rgb, img1_tensor = get_image(img1_path)
img2_rgb, img2_tensor = get_image(img2_path)

# Let's define multiple types of query points
test_points = {
    'Distinct_Object': (320, 240),      # Center, busy texture area
    'Blank_Wall': (550, 100),           # Top right empty wall corner
    'Repetitive_Keyboard': (300, 360)   # On the keyboard keys
}

gt_file = '../tum_rgb_dataset/groundtruth.txt'
poses = read_trajectory(gt_file)
T1 = get_pose_for_image(img1_path, poses)
T2 = get_pose_for_image(img2_path, poses)

if T1 is not None and T2 is not None:
    F = compute_fundamental_matrix(T1, T2, K, K)
    
    # Run test for each query point
    for pt_name, (ux, uy) in test_points.items():
        print(f"\nEvaluating: {pt_name} at ({ux}, {uy})")
        cross_attn_matrices.clear()
        
        hooks = []
        hooks.append(model.matcher.backbone.AttentionBlock1.block[2].attn.register_forward_hook(get_cross_attn('stage1_cross', model.matcher.backbone.AttentionBlock1.block[2].attn, ux, uy)))
        hooks.append(model.matcher.backbone.AttentionBlock2.block[2].attn.register_forward_hook(get_cross_attn('stage2_cross', model.matcher.backbone.AttentionBlock2.block[2].attn, ux, uy)))
        hooks.append(model.matcher.backbone.AttentionBlock3.block[2].attn.register_forward_hook(get_cross_attn('stage3_cross', model.matcher.backbone.AttentionBlock3.block[2].attn, ux, uy)))
        hooks.append(model.matcher.backbone.AttentionBlock4.block[2].attn.register_forward_hook(get_cross_attn('stage4_cross', model.matcher.backbone.AttentionBlock4.block[2].attn, ux, uy)))
        
        with torch.no_grad():
            model.matcher({'image0': img1_tensor, 'image1': img2_tensor})
        
        for h in hooks: h.remove()
        
        p = np.array([ux, uy, 1.0])
        l_prime = F @ p 
        a, b, c = l_prime
        
        # Plot only Stage 4 visualization for the ambiguous test
        attn_map = cross_attn_matrices['stage4_cross']
        H_feat, W_feat = HW_layer_map['stage4_cross']
        attn_heatmap = attn_map.reshape(H_feat, W_feat).numpy()
        attn_resized = cv2.resize(attn_heatmap, (640, 480))
        
        # Calculate EAR metric comparison
        attn_prob = attn_resized / np.sum(attn_resized)
        ear_val = compute_EAR(attn_prob, a, b, c, 10)
        baseline = random_baseline_EAR(480, 640, a, b, c, 10)
        print(f"  Stage 4 EAR Score:   {ear_val:.4f} (Baseline: {baseline:.4f})")
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        axes[0].imshow(img1_rgb)
        axes[0].plot(ux, uy, 'r*', markersize=15)
        axes[0].set_title(f"Source: {pt_name}")
        
        attn_norm = (attn_resized - attn_resized.min()) / (attn_resized.max() - attn_resized.min() + 1e-8)
        axes[1].imshow(img2_rgb)
        axes[1].imshow(attn_norm, cmap='jet', alpha=0.5)
        
        x0, x1 = 0, 640
        y0 = int(-(a*x0 + c) / b)
        y1 = int(-(a*x1 + c) / b)
        axes[1].plot([x0, x1], [y0, y1], 'w--', linewidth=2, label='Epipolar Line')
        axes[1].set_xlim([0, 640])
        axes[1].set_ylim([480, 0])
        axes[1].set_title(f"Target: Stage 4 Attention\nEAR: {ear_val:.4f}")
        plt.tight_layout()
        plt.savefig(f'ambiguous_{pt_name}.png')
        print(f"Saved ambiguous_{pt_name}.png")

    print("\nAmbiguous tests complete!")
else:
    print("Could not find matching GT poses")
