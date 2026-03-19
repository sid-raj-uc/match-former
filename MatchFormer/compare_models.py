"""
Compare 4 model variants on the same 50 pairs:
  1. Pretrained (indoor-lite-LA.ckpt) — vanilla
  2. Pretrained + epipolar mask at inference
  3. Fine-tuned 10k steps (last.ckpt) — vanilla
  4. Fine-tuned 20k steps (epipolar-model.ckpt) — vanilla
"""
import torch
import torch.nn.functional as F
import cv2
import numpy as np
import glob
import os
from tqdm import tqdm

from config.defaultmf import get_cfg_defaults
from model.lightning_loftr import PL_LoFTR
from model.backbone.coarse_matching import CoarseMatching
from model.utils.misc import lower_config
from gt_epipolar import compute_fundamental_matrix

H_img, W_img = 480, 640
stride = 8
TAU = 50.0
PAIR_GAP = 20
N_PAIRS = 50
DATA_DIR = '../data/scans/scene0011_00/exported'

# ── Epipolar mask ─────────────────────────────────────────────────────────────

def get_epipolar_mask(F_mat, H0, W0, H1, W1, tau=TAU, device='cpu'):
    y0, x0 = torch.meshgrid(torch.arange(H0), torch.arange(W0), indexing='ij')
    x0_img = (x0.float() / W0) * W_img
    y0_img = (y0.float() / H0) * H_img
    y1, x1 = torch.meshgrid(torch.arange(H1), torch.arange(W1), indexing='ij')
    x1_img = (x1.float() / W1) * W_img
    y1_img = (y1.float() / H1) * H_img
    p0 = torch.stack([x0_img.flatten(), y0_img.flatten(), torch.ones(H0*W0)], 1).to(device)
    p1 = torch.stack([x1_img.flatten(), y1_img.flatten(), torch.ones(H1*W1)], 1).to(device)
    F_t = torch.tensor(F_mat, dtype=torch.float32, device=device)
    l = p0 @ F_t.T
    num = torch.abs(l @ p1.T)
    denom = torch.sqrt(l[:,0]**2 + l[:,1]**2).unsqueeze(1)
    return torch.exp(-num / (denom + 1e-8)).unsqueeze(0)

# ── Monkey-patch ───────────────────────────────────────────────────────────────

original_forward = CoarseMatching.forward

def constrained_forward(self, feat_c0, feat_c1, data, mask_c0=None, mask_c1=None):
    N, L, S, C = feat_c0.size(0), feat_c0.size(1), feat_c1.size(1), feat_c0.size(2)
    feat_c0_n, feat_c1_n = map(lambda f: f / f.shape[-1]**.5, [feat_c0, feat_c1])
    if self.match_type == 'dual_softmax':
        sim = torch.einsum('nlc,nsc->nls', feat_c0_n, feat_c1_n) / self.temperature
        if mask_c0 is not None:
            sim.masked_fill_(~(mask_c0[...,None] * mask_c1[:,None]).bool(), -1e9)
        conf = F.softmax(sim, 1) * F.softmax(sim, 2)
        if getattr(self, 'epipolar_F', None) is not None:
            epi = get_epipolar_mask(self.epipolar_F, *data['hw0_c'], *data['hw1_c'],
                                     tau=getattr(self, 'epipolar_tau', TAU),
                                     device=conf.device)
            conf = conf * epi
    data.update({'conf_matrix': conf})
    data.update(**self.get_coarse_match(conf, data))

CoarseMatching.forward = constrained_forward

# ── Helpers ────────────────────────────────────────────────────────────────────

def get_image_tensor(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None: return None
    img = cv2.resize(img, (W_img, H_img))
    return torch.from_numpy(img).float().div(255).unsqueeze(0).unsqueeze(0)

def get_gt_matches(mkpts0, depth_path, T1, T2, K):
    depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
    if depth is None:
        return np.zeros(len(mkpts0), dtype=bool), np.zeros_like(mkpts0)
    depth = depth.astype(np.float32) / 1000.0
    fx, fy, cx, cy = K[0,0], K[1,1], K[0,2], K[1,2]
    T_cv2gl = np.array([[1,0,0,0],[0,-1,0,0],[0,0,-1,0],[0,0,0,1]], dtype=np.float64)
    T_12 = T_cv2gl @ np.linalg.inv(T2) @ T1 @ T_cv2gl
    xi = np.round(mkpts0[:,0]).astype(int)
    yi = np.round(mkpts0[:,1]).astype(int)
    valid = (xi>=0)&(xi<depth.shape[1])&(yi>=0)&(yi<depth.shape[0])
    z = np.zeros(len(mkpts0), dtype=np.float32)
    z[valid] = depth[yi[valid], xi[valid]]
    valid &= (z > 0.1) & (z <= 10.0)
    xc = (mkpts0[:,0] - cx) * z / fx
    yc = (mkpts0[:,1] - cy) * z / fy
    pts = np.stack([xc, yc, z, np.ones(len(mkpts0))], 1)
    pts2 = (T_12 @ pts.T).T
    valid &= pts2[:,2] > 0
    u2 = np.where(valid, pts2[:,0]*fx/np.where(valid,pts2[:,2],1)+cx, 0)
    v2 = np.where(valid, pts2[:,1]*fy/np.where(valid,pts2[:,2],1)+cy, 0)
    valid &= (u2>=0)&(u2<W_img)&(v2>=0)&(v2<H_img)
    return valid, np.stack([u2, v2], 1)

def build_model(ckpt_path, device):
    config = get_cfg_defaults()
    config.MATCHFORMER.BACKBONE_TYPE = 'litela'
    config.MATCHFORMER.SCENS = 'indoor'
    config.MATCHFORMER.RESOLUTION = (8, 4)
    config.MATCHFORMER.COARSE.D_MODEL = 192
    config.MATCHFORMER.COARSE.D_FFN = 192
    model = PL_LoFTR(config, pretrained_ckpt=ckpt_path)
    model.to(device).eval()
    return model

def run_pair(idx0, idx1, all_imgs, data_dir, K, model, use_epipolar, thr, device):
    n0 = os.path.basename(all_imgs[idx0]).split('.')[0]
    n1 = os.path.basename(all_imgs[idx1]).split('.')[0]
    try:
        T0 = np.loadtxt(os.path.join(data_dir, 'pose', f'{n0}.txt'))
        T1 = np.loadtxt(os.path.join(data_dir, 'pose', f'{n1}.txt'))
    except FileNotFoundError:
        return None
    if not (np.isfinite(T0).all() and np.isfinite(T1).all()): return None

    T_cv2gl = np.array([[1,0,0,0],[0,-1,0,0],[0,0,-1,0],[0,0,0,1]])
    F_mat = compute_fundamental_matrix(T0 @ T_cv2gl, T1 @ T_cv2gl, K, K)

    img0 = get_image_tensor(all_imgs[idx0])
    img1 = get_image_tensor(all_imgs[idx1])
    if img0 is None or img1 is None: return None

    model.matcher.coarse_matching.thr = thr
    model.matcher.coarse_matching.epipolar_F = F_mat if use_epipolar else None
    model.matcher.coarse_matching.epipolar_tau = TAU

    data = {'image0': img0.to(device), 'image1': img1.to(device)}
    with torch.no_grad():
        model.matcher(data)

    mkpts0 = data['mkpts0_f'].cpu().numpy()
    mkpts1 = data['mkpts1_f'].cpu().numpy()
    if len(mkpts0) == 0: return {'n': 0, 'errs': np.array([])}

    depth_path = os.path.join(data_dir, 'depth', f'{n0}.png')
    valid, gt1 = get_gt_matches(mkpts0, depth_path, T0, T1, K)
    errs = np.linalg.norm(mkpts1[valid] - gt1[valid], axis=1) if valid.sum() > 0 else np.array([])
    return {'n': len(mkpts0), 'errs': errs}


def main():
    if torch.cuda.is_available():   device = 'cuda'
    elif torch.backends.mps.is_available(): device = 'mps'
    else: device = 'cpu'

    data_dir = DATA_DIR
    K = np.loadtxt(os.path.join(data_dir, 'intrinsic', 'intrinsic_depth.txt'))[:3,:3]
    all_imgs = sorted(glob.glob(os.path.join(data_dir, 'color', '*.jpg')),
                      key=lambda x: int(os.path.basename(x).split('.')[0]))

    models_cfg = [
        ('Pretrained (vanilla)',     'model/weights/indoor-lite-LA.ckpt',          False),
        ('Pretrained + Epipolar',    'model/weights/indoor-lite-LA.ckpt',          True),
        ('Fine-tuned 10k',           'model/weights/last.ckpt',                    False),
        ('Fine-tuned 15k',           'model/weights/epipolar-step=15000.ckpt',     False),
        ('Fine-tuned 17.5k',         'model/weights/epipolar-step=17500.ckpt',     False),
        ('Fine-tuned 20k',           'model/weights/epipolar-model.ckpt',          False),
        ('Fine-tuned model3',        'model/weights/model3.ckpt',                  False),
    ]

    thr_values = [0.005, 0.01, 0.03, 0.06, 0.1, 0.2]

    # Step 1: collect valid pair indices using pretrained model at thr=0.005
    print(f"Collecting {N_PAIRS} valid pairs...")
    ref_model = build_model('model/weights/indoor-lite-LA.ckpt', device)
    valid_idxs = []
    idx = 0
    pbar = tqdm(total=N_PAIRS)
    while len(valid_idxs) < N_PAIRS and idx < len(all_imgs) - PAIR_GAP:
        res = run_pair(idx, idx+PAIR_GAP, all_imgs, data_dir, K, ref_model,
                       use_epipolar=False, thr=0.005, device=device)
        if res is not None and res['n'] > 0:
            valid_idxs.append(idx)
            pbar.update(1)
        idx += 1
    pbar.close()
    del ref_model
    print(f"Collected {len(valid_idxs)} pairs.\n")

    # Step 2: run each model config on those pairs
    header = f"{'Model':<28} | {'thr':<5} | {'Matches':>7} | {'P@3px':>7} | {'P@5px':>7} | {'MeanErr':>8}"
    sep = "─" * len(header)

    all_rows = []

    for label, ckpt, use_epi in models_cfg:
        print(f"Running: {label} ...")
        model = build_model(ckpt, device)
        for thr in thr_values:
            results = []
            for i in valid_idxs:
                r = run_pair(i, i+PAIR_GAP, all_imgs, data_dir, K, model,
                             use_epipolar=use_epi, thr=thr, device=device)
                if r is not None:
                    results.append(r)

            if not results:
                all_rows.append((label, thr, 0, 0.0, 0.0, float('nan')))
                continue

            avg_n   = np.mean([r['n'] for r in results])
            all_err = np.concatenate([r['errs'] for r in results])
            if len(all_err) == 0:
                all_rows.append((label, thr, avg_n, 0.0, 0.0, float('nan')))
                continue
            p3  = np.mean(all_err < 3.0)
            p5  = np.mean(all_err < 5.0)
            me  = np.mean(all_err)
            all_rows.append((label, thr, avg_n, p3, p5, me))
        del model

    # Print table
    print()
    print("=" * (len(header)+2))
    print("  MODEL COMPARISON  (50 pairs, tau=50.0 for epipolar)")
    print("=" * (len(header)+2))
    print(header)
    prev_label = None
    for label, thr, n, p3, p5, me in all_rows:
        if label != prev_label:
            print(sep)
            prev_label = label
        me_s = f"{me:8.2f}" if not np.isnan(me) else "      --"
        print(f"{label:<28} | {thr:<5} | {n:7.1f} | {p3:7.2%} | {p5:7.2%} | {me_s}")
    print(sep)


if __name__ == '__main__':
    with torch.no_grad():
        main()
