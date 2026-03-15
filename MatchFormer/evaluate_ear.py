import torch
import cv2
import numpy as np

# Load our utilities
from gt_epipolar import compute_fundamental_matrix, K
from match_poses import read_trajectory, get_pose_for_image
from compute_ear import compute_EAR, random_baseline_EAR

print("Testing Phase 2: Computing Epipolar Attention Ratio (EAR)")

# To compute EAR directly on the maps without re-running the model overhead right now,
# we need to modify our test_pipeline to output the maps, or just copy the tensor logic here fast. 
# Since we didn't save the raw arrays, let's load them fast. Wait, running the model again is fine on small images.

from config.defaultmf import get_cfg_defaults
from model.lightning_loftr import PL_LoFTR

config = get_cfg_defaults()
config.MATCHFORMER.BACKBONE_TYPE = 'litela'
config.MATCHFORMER.SCENS = 'indoor'
config.MATCHFORMER.RESOLUTION = (8, 4)
config.MATCHFORMER.COARSE.D_MODEL = 192
config.MATCHFORMER.COARSE.D_FFN = 192
model = PL_LoFTR(config, pretrained_ckpt='model/weights/indoor-lite-LA.ckpt')
model.eval()

img1_path = '../tum_rgb_dataset/rgb/1305031125.479424.png'
img2_path = '../tum_rgb_dataset/rgb/1305031113.411625.png'

def get_image(path):
    img = cv2.resize(cv2.imread(path, cv2.IMREAD_GRAYSCALE), (640, 480))
    return torch.from_numpy(img).float().unsqueeze(0).unsqueeze(0) / 255.0

img1_tensor = get_image(img1_path)
img2_tensor = get_image(img2_path)

cross_attn_matrices = {}

def get_cross_attn(name, layer, ux, uy, H_feat, W_feat):
    def hook(model, input, output):
        x = input[0]
        B, N, C = x.shape
        query = layer.q(x).reshape(B, N, layer.num_heads, C // layer.num_heads).permute(0, 1, 2, 3)
        kv = layer.kv(x).reshape(B, -1, 2, layer.num_heads, C // layer.num_heads).permute(2, 0, 1, 3, 4)
        
        if layer.cross:
            key = torch.cat([kv[0].split(B//2)[1], kv[0].split(B//2)[0]], dim=0)
        else:
            return 
            
        Q = layer.feature_map(query).permute(0, 2, 1, 3) 
        K = layer.feature_map(key).permute(0, 2, 1, 3)
        
        feat_x = int((ux / 640.0) * W_feat)
        feat_y = int((uy / 480.0) * H_feat)
        query_idx = feat_y * W_feat + feat_x

        Q_single = Q[0, :, query_idx:query_idx+1, :] 
        K_targets = K[0, :, :, :]
        
        attn = torch.matmul(Q_single, K_targets.transpose(-2, -1))
        Z = 1 / (torch.einsum("hd,hd->h", Q_single[:, 0, :], K_targets.sum(dim=1)) + layer.eps)
        attn = attn * Z.view(-1, 1, 1) 
        cross_attn_matrices[name] = attn.mean(dim=0).squeeze(0).detach().numpy()
    return hook

ux, uy = 320, 240
hooks = []
hooks.append(model.matcher.backbone.AttentionBlock1.block[2].attn.register_forward_hook(get_cross_attn('stage1_cross', model.matcher.backbone.AttentionBlock1.block[2].attn, ux, uy, 120, 160)))
hooks.append(model.matcher.backbone.AttentionBlock2.block[2].attn.register_forward_hook(get_cross_attn('stage2_cross', model.matcher.backbone.AttentionBlock2.block[2].attn, ux, uy, 60, 80)))
hooks.append(model.matcher.backbone.AttentionBlock3.block[2].attn.register_forward_hook(get_cross_attn('stage3_cross', model.matcher.backbone.AttentionBlock3.block[2].attn, ux, uy, 30, 40)))
hooks.append(model.matcher.backbone.AttentionBlock4.block[2].attn.register_forward_hook(get_cross_attn('stage4_cross', model.matcher.backbone.AttentionBlock4.block[2].attn, ux, uy, 15, 20)))

with torch.no_grad():
    model.matcher({'image0': img1_tensor, 'image1': img2_tensor})

for h in hooks: h.remove()

poses = read_trajectory('../tum_rgb_dataset/groundtruth.txt')
T1 = get_pose_for_image(img1_path, poses)
T2 = get_pose_for_image(img2_path, poses)

if T1 is not None and T2 is not None:
    F_matrix = compute_fundamental_matrix(T1, T2, K, K)
    l_prime = F_matrix @ np.array([ux, uy, 1.0])
    a, b, c = l_prime

    print(f"\n--- EAR (k=10 pixels) for Query Pt: ({ux}, {uy}) ---")
    k_thr = 10
    
    # Needs to match spatial coordinates. The line is calculated for 640x480 dimensions. 
    # But attention_map is shape H_feat x W_feat. We need to scale k, a, b, c to feature dimensions, 
    # OR we need to resize the attention map to 640x480. 
    # Resizing attention map to 640x480 is easier and aligns closely to the metrics visual.
    
    for name, attn_row in cross_attn_matrices.items():
        if name == 'stage1_cross': h_f, w_f = 120, 160
        elif name == 'stage2_cross': h_f, w_f = 60, 80
        elif name == 'stage3_cross': h_f, w_f = 30, 40
        else: h_f, w_f = 15, 20
        
        attn_map = attn_row.reshape(h_f, w_f)
        attn_resized = cv2.resize(attn_map, (640, 480))
        
        # In EAR matching, resizing means we need to re-normalize the probabilities so they sum to 1.0 again
        attn_resized = attn_resized / np.sum(attn_resized)
        
        ear_val = compute_EAR(attn_resized, a, b, c, k_thr)
        baseline = random_baseline_EAR(480, 640, a, b, c, k_thr)
        
        print(f"Layer: {name}")
        print(f"  EAR Score:   {ear_val:.4f}")
        print(f"  Random Base: {baseline:.4f}")
        print("-" * 30)

