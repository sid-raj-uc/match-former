import torch
import cv2
import numpy as np

# Load our utilities
from config.defaultmf import get_cfg_defaults
from model.lightning_loftr import PL_LoFTR
from match_poses import read_trajectory, get_pose_for_image
from gt_epipolar import compute_fundamental_matrix, K
from plot_epipolar_attention import plot_epipolar_attention

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

# Query point in img1
ux, uy = 320, 240 # Center of 640x480

HW_layer_map = {
   'stage1_cross': (120, 160), # 480/4 = 120, 640/4 = 160
   'stage2_cross': (60, 80),   # /8
   'stage3_cross_1': (30, 40), # /16
   'stage3_cross_2': (30, 40), # /16
   'stage4_cross_1': (15, 20), # /32
   'stage4_cross_2': (15, 20)  # /32
}

def get_cross_attn(name, layer):
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
        
        # Optimize memory footprint by not constructing the full O(N^2) Matrix
        # Instead just process the point at ux, uy map coordinates.
        H_feat, W_feat = HW_layer_map[name]
        feat_x = int((ux / 640.0) * W_feat)
        feat_y = int((uy / 480.0) * H_feat)
        query_idx = feat_y * W_feat + feat_x

        # Get only the queried row:
        # B=2: batch[0] is img1->img2, batch[1] is img2->img1. 
        # So we only need the first batch and specific query row.
        Q_single = Q[0, :, query_idx:query_idx+1, :] # Shape: (num_heads, 1, head_dim)
        K_targets = K[0, :, :, :] # Shape: (num_heads, N, head_dim)
        
        # attn for one row: (num_heads, 1, head_dim) @ (num_heads, head_dim, N) -> (num_heads, 1, N)
        attn = torch.matmul(Q_single, K_targets.transpose(-2, -1))
        
        # Normalization factor Z (for the single query index)
        # Note: Summing early prevents big matrices.
        Z = 1 / (torch.einsum("hd,hd->h", Q_single[:, 0, :], K_targets.sum(dim=1)) + layer.eps)
        
        # apply normalization
        attn = attn * Z.view(-1, 1, 1) # -> (num_heads, 1, N)
        
        # Remove head dim by averaging -> (1, N) and save
        cross_attn_matrices[name] = attn.mean(dim=0).squeeze(0).detach().cpu()
        
    return hook

hook_handles_attn = []
hook_handles_attn.append(model.matcher.backbone.AttentionBlock1.block[2].attn.register_forward_hook(
    get_cross_attn('stage1_cross', model.matcher.backbone.AttentionBlock1.block[2].attn)))
hook_handles_attn.append(model.matcher.backbone.AttentionBlock2.block[2].attn.register_forward_hook(
    get_cross_attn('stage2_cross', model.matcher.backbone.AttentionBlock2.block[2].attn)))
hook_handles_attn.append(model.matcher.backbone.AttentionBlock3.block[1].attn.register_forward_hook(
    get_cross_attn('stage3_cross_1', model.matcher.backbone.AttentionBlock3.block[1].attn)))
hook_handles_attn.append(model.matcher.backbone.AttentionBlock3.block[2].attn.register_forward_hook(
    get_cross_attn('stage3_cross_2', model.matcher.backbone.AttentionBlock3.block[2].attn)))
hook_handles_attn.append(model.matcher.backbone.AttentionBlock4.block[1].attn.register_forward_hook(
    get_cross_attn('stage4_cross_1', model.matcher.backbone.AttentionBlock4.block[1].attn)))
hook_handles_attn.append(model.matcher.backbone.AttentionBlock4.block[2].attn.register_forward_hook(
    get_cross_attn('stage4_cross_2', model.matcher.backbone.AttentionBlock4.block[2].attn)))

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

print("Running model (Low Memory hooks)...")
input_data = {'image0': img1_tensor, 'image1': img2_tensor}
with torch.no_grad():
    model.matcher(input_data)

for handle in hook_handles_attn:
    handle.remove()

# Get Poses and compute F
gt_file = '../tum_rgb_dataset/groundtruth.txt'
poses = read_trajectory(gt_file)
T1 = get_pose_for_image(img1_path, poses)
T2 = get_pose_for_image(img2_path, poses)

if T1 is not None and T2 is not None:
    F = compute_fundamental_matrix(T1, T2, K, K)
    print("Found GT Poses and F matrix!")

    attention_heatmaps = []
    layer_names = []
    
    # Collect saved per-layer values
    for name, attn_row in cross_attn_matrices.items():
        H_feat, W_feat = HW_layer_map[name]
        
        # Reshape to feature map shape 
        attn_heatmap = attn_row.reshape(H_feat, W_feat).numpy()
        
        attention_heatmaps.append(attn_heatmap)
        layer_names.append(name)
        print(f"Computed attention for {name}. Max val: {attn_heatmap.max():.4f}")

    print("Heatmap generation complete. Saving sample plot...")
    
    import matplotlib.pyplot as plt
    
    p = np.array([ux, uy, 1.0])
    l_prime = F @ p 
    a, b, c = l_prime
    
    num_layers = len(attention_heatmaps)
    cols = 3
    rows = (num_layers + 1) // cols + 1
    
    fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))
    axes = axes.flatten()
    axes[0].imshow(img1_rgb)
    axes[0].plot(ux, uy, 'r*', markersize=15)
    axes[0].set_title("Source Image (Query Point)")
    axes[0].axis('off')
    
    h, w = img2_rgb.shape[:2]
    
    for i, (attn_map, name) in enumerate(zip(attention_heatmaps, layer_names)):
        ax = axes[i + 1]
        attn_resized = cv2.resize(attn_map, (w, h))
        attn_resized = (attn_resized - attn_resized.min()) / (attn_resized.max() - attn_resized.min() + 1e-8)
        
        ax.imshow(img2_rgb)
        ax.imshow(attn_resized, cmap='jet', alpha=0.5)
        
        x0, x1 = 0, w
        y0 = int(-(a*x0 + c) / b)
        y1 = int(-(a*x1 + c) / b)
        ax.plot([x0, x1], [y0, y1], 'w--', linewidth=2, label='Epipolar Line')
        
        # Set tight axis limits to image bounds
        ax.set_xlim([0, w])
        ax.set_ylim([h, 0])
        ax.set_title(f"Target Image - {name}")
        ax.axis('off')

    for j in range(i + 2, len(axes)):
        axes[j].axis('off')
        
    plt.tight_layout()
    plt.savefig('epipolar_attention_sample.png')
    print("Saved sample to epipolar_attention_sample.png")
else:
    print("Could not find matching GT poses")
