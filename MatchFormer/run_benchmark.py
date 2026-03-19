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
    depth = cv2.imread(depth1_path, cv2.IMREAD_UNCHANGED)
    if depth is None: return np.zeros(len(mkpts0), dtype=bool), np.zeros_like(mkpts0)
    depth = depth.astype(np.float32) / 1000.0

    fx, fy, cx, cy = K[0,0], K[1,1], K[0,2], K[1,2]
    T_cv2gl = np.array([[1,0,0,0],[0,-1,0,0],[0,0,-1,0],[0,0,0,1]], dtype=np.float64)
    T_12 = T_cv2gl @ np.linalg.inv(T2) @ T1 @ T_cv2gl  # compute once

    x_idx = np.round(mkpts0[:, 0]).astype(int)
    y_idx = np.round(mkpts0[:, 1]).astype(int)

    valid = (x_idx >= 0) & (x_idx < depth.shape[1]) & (y_idx >= 0) & (y_idx < depth.shape[0])
    z = np.zeros(len(mkpts0), dtype=np.float32)
    z[valid] = depth[y_idx[valid], x_idx[valid]]
    valid &= (z > 0.1) & (z <= 10.0)

    x_c = (mkpts0[:, 0] - cx) * z / fx
    y_c = (mkpts0[:, 1] - cy) * z / fy
    pts_h = np.stack([x_c, y_c, z, np.ones(len(mkpts0))], axis=1)  # [N, 4]
    pts_c2 = (T_12 @ pts_h.T).T  # [N, 4]

    valid &= pts_c2[:, 2] > 0
    u2 = np.where(valid, pts_c2[:, 0] * fx / np.where(valid, pts_c2[:, 2], 1) + cx, 0)
    v2 = np.where(valid, pts_c2[:, 1] * fy / np.where(valid, pts_c2[:, 2], 1) + cy, 0)
    valid &= (u2 >= 0) & (u2 < W_img) & (v2 >= 0) & (v2 < H_img)

    gt_pts = np.stack([u2, v2], axis=1)
    return valid, gt_pts

def evaluate_pair(img0_idx, img1_idx, all_imgs, data_dir, K, model, tau_values, device='cpu', debug=False):
    import time
    def t(msg):
        if debug: print(f'  [{time.time():.3f}] {msg}', flush=True)
    path0 = all_imgs[img0_idx]
    path1 = all_imgs[img1_idx]

    img0_idx_num = os.path.basename(path0).split('.')[0]
    img1_idx_num = os.path.basename(path1).split('.')[0]

    try:
        T0 = np.loadtxt(os.path.join(data_dir, 'pose', f'{img0_idx_num}.txt'))
        T1 = np.loadtxt(os.path.join(data_dir, 'pose', f'{img1_idx_num}.txt'))
    except FileNotFoundError: return None
    t('poses loaded')

    if not np.isfinite(T0).all() or not np.isfinite(T1).all(): return None

    T_cv2gl = np.array([[1,0,0,0],[0,-1,0,0],[0,0,-1,0],[0,0,0,1]])
    T0_cv = T0 @ T_cv2gl
    T1_cv = T1 @ T_cv2gl
    F_mat = compute_fundamental_matrix(T0_cv, T1_cv, K, K)
    t('F_mat computed')

    img0 = get_image_tensor(path0)
    img1 = get_image_tensor(path1)
    if img0 is None or img1 is None: return None
    t('images loaded')

    depth0_path = os.path.join(data_dir, 'depth', f'{img0_idx_num}.png')
    input_data = {'image0': img0.to(device), 'image1': img1.to(device)}

    # Run vanilla once
    model.matcher.coarse_matching.epipolar_F = None
    t('starting vanilla forward pass')
    with torch.no_grad():
        model.matcher(input_data)
    t('vanilla forward pass done')
    mkpts0_v = input_data['mkpts0_f'].cpu().numpy()
    mkpts1_v = input_data['mkpts1_f'].cpu().numpy()
    t(f'vanilla: {len(mkpts0_v)} matches')

    valid_mask_v, gt_mkpts1_v = get_gt_matches(mkpts0_v, depth0_path, T0, T1, K)
    mkpts0_v = mkpts0_v[valid_mask_v]
    mkpts1_v = mkpts1_v[valid_mask_v]
    gt_mkpts1_v = gt_mkpts1_v[valid_mask_v]
    t(f'gt_matches done: {len(mkpts0_v)} valid')
    if len(mkpts0_v) == 0: return None

    errs_v = compute_gt_errors(mkpts1_v, gt_mkpts1_v)
    vanilla = {
        'v_total': len(errs_v),
        'v_mean_err': np.mean(errs_v) if len(errs_v) > 0 else 0,
        'v_p3': np.mean(errs_v < 3.0) if len(errs_v) > 0 else 0,
        'v_p5': np.mean(errs_v < 5.0) if len(errs_v) > 0 else 0,
    }

    # Sweep all taus in one pass per pair
    results_by_tau = {}
    for tau in tau_values:
        t(f'starting constrained forward pass tau={tau}')
        model.matcher.coarse_matching.epipolar_F = F_mat
        model.matcher.coarse_matching.epipolar_tau = tau
        with torch.no_grad():
            model.matcher(input_data)
        t(f'constrained forward pass done tau={tau}')
        mkpts0_c = input_data['mkpts0_f'].cpu().numpy()
        mkpts1_c = input_data['mkpts1_f'].cpu().numpy()
        valid_mask_c, gt_mkpts1_c = get_gt_matches(mkpts0_c, depth0_path, T0, T1, K)
        mkpts1_c = mkpts1_c[valid_mask_c]
        gt_mkpts1_c = gt_mkpts1_c[valid_mask_c]
        errs_c = compute_gt_errors(mkpts1_c, gt_mkpts1_c)
        results_by_tau[tau] = {
            'c_total': len(errs_c),
            'c_mean_err': np.mean(errs_c) if len(errs_c) > 0 else 0,
            'c_p3': np.mean(errs_c < 3.0) if len(errs_c) > 0 else 0,
            'c_p5': np.mean(errs_c < 5.0) if len(errs_c) > 0 else 0,
        }

    return {'vanilla': vanilla, 'by_tau': results_by_tau}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_pairs', type=int, default=100)
    parser.add_argument('--ckpt', type=str, default='model/weights/indoor-lite-LA.ckpt')
    parser.add_argument('--data_dir', type=str, default='../data/scans/scene0000_00/exported')
    parser.add_argument('--test_only', action='store_true',
                        help='Evaluate on last 10%% of frames only (held-out test split)')
    args = parser.parse_args()

    # Model Setup
    config = get_cfg_defaults()
    config.MATCHFORMER.BACKBONE_TYPE = 'litela'
    config.MATCHFORMER.SCENS = 'indoor'
    config.MATCHFORMER.RESOLUTION = (8, 4)
    config.MATCHFORMER.COARSE.D_MODEL = 192
    config.MATCHFORMER.COARSE.D_FFN = 192

    if torch.cuda.is_available():
        device = 'cuda'
    elif torch.backends.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'
    model = PL_LoFTR(config, pretrained_ckpt=args.ckpt)
    model.to(device)
    model.eval()

    # Data Setup
    data_dir = args.data_dir
    color_dir = os.path.join(data_dir, 'color', '*.jpg')
    all_imgs = sorted(glob.glob(color_dir), key=lambda x: int(os.path.basename(x).split('.')[0]))

    if args.test_only:
        n_test = max(1, int(len(all_imgs) * 0.1))
        all_imgs = all_imgs[-n_test:]
        print(f'Test-only mode: using last {len(all_imgs)} frames')

    K = np.loadtxt(os.path.join(data_dir, 'intrinsic', 'intrinsic_depth.txt'))[:3, :3]
    
    thr_values = [0.005, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.1, 0.2]
    tau = 50.0  # fixed tau
    pair_gap = 20
    results = []
    idx = 0

    print(f"--- Threshold Sweep Benchmark: {args.num_pairs} Pairs, tau={tau} ---")
    print(f"Sweeping thresholds: {thr_values}\n")

    # Step 1: Collect valid image index pairs at lowest threshold
    model.matcher.coarse_matching.thr = thr_values[0]
    valid_idxs = []
    pbar = tqdm(total=args.num_pairs, desc='Collecting pairs', unit='pair')
    while len(valid_idxs) < args.num_pairs and idx < len(all_imgs) - pair_gap:
        res = evaluate_pair(idx, idx + pair_gap, all_imgs, data_dir, K, model, [tau], device=device, debug=(idx==0))
        if res is not None:
            valid_idxs.append(idx)
            pbar.set_postfix({'pairs': len(valid_idxs)})
            pbar.update(1)
        idx += 1
    pbar.close()

    if not valid_idxs:
        print("No valid pairs found even at lowest threshold.")
        return

    print(f"\nCollected {len(valid_idxs)} valid pairs at thr={thr_values[0]}")

    # Step 2: Sweep thresholds on the same pairs
    print("\n" + "=" * 65)
    print("THRESHOLD SWEEP RESULTS  (tau=50.0 fixed)")
    print("=" * 65)
    print(f"{'Thr':<8} | {'Mean Err (px)':<15} | {'P@3px':<10} | {'P@5px':<10} | {'Avg Matches'}")
    print("-" * 65)

    for thr in thr_values:
        model.matcher.coarse_matching.thr = thr
        thr_results = []
        for i in valid_idxs:
            res = evaluate_pair(i, i + pair_gap, all_imgs, data_dir, K, model, [tau], device=device)
            if res is not None:
                thr_results.append(res)

        if not thr_results:
            print(f"{thr:<8} | no matches")
            continue

        mean_err = np.mean([r['vanilla']['v_mean_err'] for r in thr_results])
        p3       = np.mean([r['vanilla']['v_p3']       for r in thr_results])
        p5       = np.mean([r['vanilla']['v_p5']       for r in thr_results])
        avg_m    = np.mean([r['vanilla']['v_total']    for r in thr_results])
        print(f"{thr:<8} | {mean_err:<15.2f} | {p3:.2%}   | {p5:.2%}   | {avg_m:.1f}")

    print("=" * 65)

if __name__ == '__main__':
    with torch.no_grad():
        main()
