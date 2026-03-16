import torch
import torch.nn.functional as F
import cv2
import numpy as np
import glob
import os
import argparse
from tqdm import tqdm

from config.defaultmf import get_cfg_defaults
from model.lightning_loftr import PL_LoFTR
from model.backbone.coarse_matching import CoarseMatching
from gt_epipolar import compute_fundamental_matrix

# Constants
H_img, W_img = 480, 640
stride = 8
H_feat, W_feat = H_img // stride, W_img // stride

def get_epipolar_mask_matrix(F_mat, H0, W0, H1, W1, H_img=480, W_img=640, tau=10.0, device='cpu'):
    y0_feat, x0_feat = torch.meshgrid(torch.arange(H0), torch.arange(W0), indexing='ij')
    x0_img = (x0_feat.float() / W0) * W_img
    y0_img = (y0_feat.float() / H0) * H_img
    
    y1_feat, x1_feat = torch.meshgrid(torch.arange(H1), torch.arange(W1), indexing='ij')
    x1_img = (x1_feat.float() / W1) * W_img
    y1_img = (y1_feat.float() / H1) * H_img
    
    p0 = torch.stack([x0_img.flatten(), y0_img.flatten(), torch.ones_like(x0_img.flatten())], dim=1).to(device)
    p1 = torch.stack([x1_img.flatten(), y1_img.flatten(), torch.ones_like(x1_img.flatten())], dim=1).to(device)
    
    F_t = torch.tensor(F_mat, dtype=torch.float32, device=device)
    l_prime = p0 @ F_t.T  
    
    num = torch.abs(l_prime @ p1.T) 
    denom = torch.sqrt(l_prime[:, 0]**2 + l_prime[:, 1]**2).unsqueeze(1) 
    
    distances = num / (denom + 1e-8)
    mask = torch.exp(-distances / tau)
    return mask.unsqueeze(0) 

# Monkey Patch Forward Pass
original_forward = CoarseMatching.forward

def constrained_forward(self, feat_c0, feat_c1, data, mask_c0=None, mask_c1=None):
    N, L, S, C = feat_c0.size(0), feat_c0.size(1), feat_c1.size(1), feat_c0.size(2)
    feat_c0_norm, feat_c1_norm = map(lambda feat: feat / (feat.shape[-1]**.5), [feat_c0, feat_c1])
    
    if self.match_type == 'dual_softmax':
        sim_matrix = torch.einsum("nlc,nsc->nls", feat_c0_norm, feat_c1_norm) / self.temperature
        if mask_c0 is not None:
             sim_matrix.masked_fill_(~(mask_c0[..., None] * mask_c1[:, None]).bool(), -1e9)
        conf_matrix = F.softmax(sim_matrix, 1) * F.softmax(sim_matrix, 2)
        
        if getattr(self, 'epipolar_F', None) is not None:
            tau = getattr(self, 'epipolar_tau', 10.0)
            H0, W0 = data['hw0_c']
            H1, W1 = data['hw1_c']
            epi_mask = get_epipolar_mask_matrix(self.epipolar_F, H0, W0, H1, W1, H_img, W_img, tau=tau, device=conf_matrix.device)
            conf_matrix = conf_matrix * epi_mask
            
    data.update({'conf_matrix': conf_matrix})
    data.update(**self.get_coarse_match(conf_matrix, data))

CoarseMatching.forward = constrained_forward


def get_image_tensor(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None: return None
    img = cv2.resize(img, (W_img, H_img))
    img_tensor = torch.from_numpy(img).float() / 255.0
    return img_tensor.unsqueeze(0).unsqueeze(0)

def compute_gt_errors(mkpts1, gt_mkpts1):
    if len(mkpts1) == 0: return np.array([])
    return np.linalg.norm(mkpts1 - gt_mkpts1, axis=1)

def get_gt_matches(mkpts0, depth1_path, T1, T2, K):
    """
    Project points from image 1 to image 2 based on depth.
    Return mask of valid matched points and their coordinates.
    """
    depth = cv2.imread(depth1_path, cv2.IMREAD_UNCHANGED)
    if depth is None: return np.zeros(len(mkpts0), dtype=bool), np.zeros_like(mkpts0)
    
    # Scale depth identical to projection map
    depth = (depth.astype(float) / 1000.0)
    
    valid_mask = np.zeros(len(mkpts0), dtype=bool)
    gt_pts = np.zeros_like(mkpts0)
    for i, pt in enumerate(mkpts0):
        # Sample Depth from Nearest Neighbor (ensure bounds)
        x_idx, y_idx = int(round(pt[0])), int(round(pt[1]))
        if y_idx >= depth.shape[0] or x_idx >= depth.shape[1] or y_idx < 0 or x_idx < 0: continue
            
        z = depth[y_idx, x_idx]
        if z <= 0.1 or z > 10.0: continue # Invalid Depth Range
        
        # 3D projection
        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]
        x_c1 = (pt[0] - cx) * z / fx
        y_c1 = (pt[1] - cy) * z / fy
        p_c1 = np.array([x_c1, y_c1, z, 1.0])
        
        # Transform
        T_cv2gl = np.array([[1,0,0,0],[0,-1,0,0],[0,0,-1,0],[0,0,0,1]])
        T_12 = T_cv2gl @ np.linalg.inv(T2) @ T1 @ T_cv2gl
        p_c2 = T_12 @ p_c1
        if p_c2[2] <= 0: continue # Behind camera 2
            
        u2 = (p_c2[0] * fx / p_c2[2]) + cx
        v2 = (p_c2[1] * fy / p_c2[2]) + cy
        
        if 0 <= u2 < W_img and 0 <= v2 < H_img:
            valid_mask[i] = True
            gt_pts[i] = [u2, v2]
            
    return valid_mask, gt_pts

def evaluate_pair(img0_idx, img1_idx, all_imgs, data_dir, K, model, tau):
    path0 = all_imgs[img0_idx]
    path1 = all_imgs[img1_idx]
    
    img0_idx_num = os.path.basename(path0).split('.')[0]
    img1_idx_num = os.path.basename(path1).split('.')[0]
    
    try:
        T0 = np.loadtxt(os.path.join(data_dir, 'pose', f'{img0_idx_num}.txt'))
        T1 = np.loadtxt(os.path.join(data_dir, 'pose', f'{img1_idx_num}.txt'))
    except FileNotFoundError: return None
    
    if not np.isfinite(T0).all() or not np.isfinite(T1).all(): return None
    
    # Map from OpenGL format to OpenCV
    T_cv2gl = np.array([[1,0,0,0],[0,-1,0,0],[0,0,-1,0],[0,0,0,1]])
    T0_cv = T0 @ T_cv2gl
    T1_cv = T1 @ T_cv2gl
    
    F_mat = compute_fundamental_matrix(T0_cv, T1_cv, K, K)
    
    img0 = get_image_tensor(path0)
    img1 = get_image_tensor(path1)
    if img0 is None or img1 is None: return None
    
    depth0_path = os.path.join(data_dir, 'depth', f'{img0_idx_num}.png')
    
    input_data = {'image0': img0, 'image1': img1}
    
    # Evaluate Validation Mask BEFORE constraint vs Vanilla.
    # To be perfectly fair, we only evaluate epipolar error on matches where a GT Point EXISTS
    # Meaning, the scene overlaps and depth is valid. 
    # If the network predicts a match for a ray that has no geometry, it's out of bounds and ambiguous. 
    
    # 1. RUN VANILLA
    model.matcher.coarse_matching.epipolar_F = None
    with torch.no_grad():
        model.matcher(input_data)
        mkpts0_v = input_data['mkpts0_f'].cpu().numpy()
        mkpts1_v = input_data['mkpts1_f'].cpu().numpy()
        
    valid_mask_v, gt_mkpts1_v = get_gt_matches(mkpts0_v, depth0_path, T0, T1, K)
    mkpts0_v, mkpts1_v, gt_mkpts1_v = mkpts0_v[valid_mask_v], mkpts1_v[valid_mask_v], gt_mkpts1_v[valid_mask_v]
    
    # Skip pairs with zero valid matches across both
    if len(mkpts0_v) == 0: return None
        
    # 2. RUN CONSTRAINED
    model.matcher.coarse_matching.epipolar_F = F_mat
    model.matcher.coarse_matching.epipolar_tau = tau
    with torch.no_grad():
        model.matcher(input_data)
        mkpts0_c = input_data['mkpts0_f'].cpu().numpy()
        mkpts1_c = input_data['mkpts1_f'].cpu().numpy()
        
    valid_mask_c, gt_mkpts1_c = get_gt_matches(mkpts0_c, depth0_path, T0, T1, K)
    mkpts0_c, mkpts1_c, gt_mkpts1_c = mkpts0_c[valid_mask_c], mkpts1_c[valid_mask_c], gt_mkpts1_c[valid_mask_c]
    
    errs_v = compute_gt_errors(mkpts1_v, gt_mkpts1_v)
    errs_c = compute_gt_errors(mkpts1_c, gt_mkpts1_c)
    
    return {
        'v_total': len(errs_v),
        'v_mean_err': np.mean(errs_v) if len(errs_v) > 0 else 0,
        'v_p3': np.mean(errs_v < 3.0) if len(errs_v) > 0 else 0,
        'v_p5': np.mean(errs_v < 5.0) if len(errs_v) > 0 else 0,
        'c_total': len(errs_c),
        'c_mean_err': np.mean(errs_c) if len(errs_c) > 0 else 0,
        'c_p3': np.mean(errs_c < 3.0) if len(errs_c) > 0 else 0,
        'c_p5': np.mean(errs_c < 5.0) if len(errs_c) > 0 else 0,
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_pairs', type=int, default=100)
    parser.add_argument('--ckpt', type=str, default='model/weights/indoor-lite-LA.ckpt')
    args = parser.parse_args()
    
    # Model Setup
    config = get_cfg_defaults()
    config.MATCHFORMER.BACKBONE_TYPE = 'litela'
    config.MATCHFORMER.SCENS = 'indoor'
    config.MATCHFORMER.RESOLUTION = (8, 4)
    config.MATCHFORMER.COARSE.D_MODEL = 192
    config.MATCHFORMER.COARSE.D_FFN = 192

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = PL_LoFTR(config, pretrained_ckpt=args.ckpt)
    model.to(device)
    model.eval()

    # Data Setup
    data_dir = '../data/scans/scene0000_00/exported'
    color_dir = os.path.join(data_dir, 'color', '*.jpg')
    all_imgs = sorted(glob.glob(color_dir), key=lambda x: int(os.path.basename(x).split('.')[0]))

    K = np.loadtxt(os.path.join(data_dir, 'intrinsic', 'intrinsic_depth.txt'))[:3, :3]
    
    tau_values = [50.0, 20.0, 10.0, 5.0, 2.0]
    
    results_by_tau = {}
    
    print(f"--- Running ScanNet Benchmark for {args.num_pairs} Pairs ---")
    
    # To make sure we hit 100 pairs, we'll keep iterating until we collect enough valid ones
    # A valid pair is one separated by at least 20 frames that has overlapping geometry
    
    for tau in tau_values:
        print(f"\nEvaluating Tau={tau}...")
        results = []
        pair_gap = 20
        idx = 0
        
        pbar = tqdm(total=args.num_pairs)
        while len(results) < args.num_pairs and idx < len(all_imgs) - pair_gap:
            res = evaluate_pair(idx, idx + pair_gap, all_imgs, data_dir, K, model, tau)
            if res is not None:
                results.append(res)
                pbar.update(1)
                
            idx += 1
            if idx >= len(all_imgs) - pair_gap:
                print("Warning: Ran out of images before reaching pair quota.")
                break
        pbar.close()
        
        v_mean_err = np.mean([r['v_mean_err'] for r in results])
        v_p3 = np.mean([r['v_p3'] for r in results])
        v_p5 = np.mean([r['v_p5'] for r in results])
        v_tot = np.mean([r['v_total'] for r in results])

        c_mean_err = np.mean([r['c_mean_err'] for r in results])
        c_p3 = np.mean([r['c_p3'] for r in results])
        c_p5 = np.mean([r['c_p5'] for r in results])
        c_tot = np.mean([r['c_total'] for r in results])
        
        results_by_tau[tau] = {
            'v_mean_err': v_mean_err, 'v_p3': v_p3, 'v_p5': v_p5, 'v_tot': v_tot,
            'c_mean_err': c_mean_err, 'c_p3': c_p3, 'c_p5': c_p5, 'c_tot': c_tot
        }
        
    print("\n" + "="*50)
    print("FINAL BENCHMARK SUMMARY")
    print("="*50)
    # The vanilla performance shouldn't change with tau, just take the first
    v_perf = results_by_tau[tau_values[0]]
    print(f"VANILLA MODEL:")
    print(f"Mean GT Error:   {v_perf['v_mean_err']:.2f} px")
    print(f"Precision @ 3px: {v_perf['v_p3']:.2%}")
    print(f"Precision @ 5px: {v_perf['v_p5']:.2%}")
    print(f"Avg Matches:     {v_perf['v_tot']:.1f}")
    print("-" * 60)
    print("CONSTRAINED MODEL PERFORMANCE SWEEP:")
    print(f"{'Tau':<8} | {'Mean Err (px)':<15} | {'P@3px':<10} | {'P@5px':<10} | {'Avg Matches'}")
    print("-" * 60)
    for tau in tau_values:
        perf = results_by_tau[tau]
        print(f"{tau:<8} | {perf['c_mean_err']:<15.2f} | {perf['c_p3']:.2%}   | {perf['c_p5']:.2%}   | {perf['c_tot']:.1f}")
    print("=" * 60)

if __name__ == '__main__':
    with torch.no_grad():
        main()
