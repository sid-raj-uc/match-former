import nbformat as nbf
nb = nbf.v4.new_notebook()
cells = []

cells.append(nbf.v4.new_markdown_cell("# Phase 4: Benchmark Visual Analysis\nComparing match precision between Vanilla MatchFormer and Epipolar-Constrained MatchFormer."))
cells.append(nbf.v4.new_code_cell("""\
%matplotlib inline
import torch
import torch.nn.functional as F
import cv2
import numpy as np
import matplotlib.pyplot as plt
import glob

from config.defaultmf import get_cfg_defaults
from model.lightning_loftr import PL_LoFTR
from model.backbone.coarse_matching import CoarseMatching
from match_poses import read_trajectory, get_pose_for_image
from gt_epipolar import compute_fundamental_matrix, K

# --- EPIPOLAR MASK ---
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
    l_prime = p0 @ F_t.T  # (N0, 3)
    
    num = torch.abs(l_prime @ p1.T) # (N0, N1)
    denom = torch.sqrt(l_prime[:, 0]**2 + l_prime[:, 1]**2).unsqueeze(1) # (N0, 1)
    
    distances = num / (denom + 1e-8)
    mask = torch.exp(-distances / tau)
    return mask.unsqueeze(0) # (1, N0, N1)

# Monkey Patch
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
            H0, W0 = data['hw0_c']
            H1, W1 = data['hw1_c']
            epi_mask = get_epipolar_mask_matrix(self.epipolar_F, H0, W0, H1, W1, 480, 640, tau=10.0, device=conf_matrix.device)
            conf_matrix = conf_matrix * epi_mask
            
    data.update({'conf_matrix': conf_matrix})
    data.update(**self.get_coarse_match(conf_matrix, data))

CoarseMatching.forward = constrained_forward

config = get_cfg_defaults()
config.MATCHFORMER.BACKBONE_TYPE = 'litela'
config.MATCHFORMER.SCENS = 'indoor'
config.MATCHFORMER.RESOLUTION = (8, 4)
config.MATCHFORMER.COARSE.D_MODEL = 192
config.MATCHFORMER.COARSE.D_FFN = 192

model = PL_LoFTR(config, pretrained_ckpt='model/weights/indoor-lite-LA.ckpt')
model.eval()

rgb_dir = '../tum_rgb_dataset/rgb/*.png'
all_imgs = sorted(glob.glob(rgb_dir))
gt_file = '../tum_rgb_dataset/groundtruth.txt'
poses = read_trajectory(gt_file)

def get_images(path0, path1):
    i0 = cv2.resize(cv2.imread(path0, cv2.IMREAD_GRAYSCALE), (640, 480))
    i1 = cv2.resize(cv2.imread(path1, cv2.IMREAD_GRAYSCALE), (640, 480))
    t0 = torch.from_numpy(i0).float() / 255.0
    t1 = torch.from_numpy(i1).float() / 255.0
    rgb0 = cv2.cvtColor(cv2.imread(path0), cv2.COLOR_BGR2RGB)
    rgb0 = cv2.resize(rgb0, (640, 480))
    rgb1 = cv2.cvtColor(cv2.imread(path1), cv2.COLOR_BGR2RGB)
    rgb1 = cv2.resize(rgb1, (640, 480))
    return t0.unsqueeze(0).unsqueeze(0), t1.unsqueeze(0).unsqueeze(0), rgb0, rgb1

def compute_epipolar_errors(mkpts0, mkpts1, F):
    if len(mkpts0) == 0: return np.array([])
    p0 = np.concatenate([mkpts0, np.ones((len(mkpts0), 1))], axis=1)
    p1 = np.concatenate([mkpts1, np.ones((len(mkpts1), 1))], axis=1)
    l_prime = p0 @ F.T 
    num = np.abs(np.sum(l_prime * p1, axis=1))
    denom = np.sqrt(l_prime[:, 0]**2 + l_prime[:, 1]**2)
    return num / (denom + 1e-8)

def plot_benchmark_triplet(img0_idx, img1_idx):
    path0 = all_imgs[img0_idx]
    path1 = all_imgs[img1_idx]
    
    T0 = get_pose_for_image(path0, poses)
    T1 = get_pose_for_image(path1, poses)
    if T0 is None or T1 is None: return
    F_mat = compute_fundamental_matrix(T0, T1, K, K)
    
    t0, t1, rgb0, rgb1 = get_images(path0, path1)
    
    # Vanilla
    model.matcher.coarse_matching.epipolar_F = None
    with torch.no_grad():
        data_v = {'image0': t0, 'image1': t1}
        model.matcher(data_v)
        m0_v = data_v['mkpts0_f'].cpu().numpy()
        m1_v = data_v['mkpts1_f'].cpu().numpy()
        
    # Constrained
    model.matcher.coarse_matching.epipolar_F = F_mat
    with torch.no_grad():
        data_c = {'image0': t0, 'image1': t1}
        model.matcher(data_c)
        m0_c = data_c['mkpts0_f'].cpu().numpy()
        m1_c = data_c['mkpts1_f'].cpu().numpy()
        
    errs_v = compute_epipolar_errors(m0_v, m1_v, F_mat)
    errs_c = compute_epipolar_errors(m0_c, m1_c, F_mat)
    
    fig, axes = plt.subplots(1, 3, figsize=(24, 6))
    
    # Plot 1: Standard Source Image
    axes[0].imshow(rgb0)
    # We'll just scatter the source points of the constrained model to keep it clean
    # Or plot nothing if you just want to see the image layout.
    axes[0].scatter(m0_c[:, 0], m0_c[:, 1], c='b', s=20, alpha=0.8, marker='.')
    axes[0].set_title(f"Source Image (img {img0_idx})\\nTotal Source Keypoints found: {len(m0_c)}")
    axes[0].axis('off')
    
    # Plot Vanilla Target Points
    axes[1].imshow(rgb1)
    good_v = errs_v < 5.0
    axes[1].scatter(m1_v[good_v, 0], m1_v[good_v, 1], c='g', s=5, alpha=0.5, label='< 5px Error')
    axes[1].scatter(m1_v[~good_v, 0], m1_v[~good_v, 1], c='r', s=5, alpha=0.5, label='> 5px Error')
    axes[1].set_title(f"MatchFormer (VANILLA)\\nTotal: {len(m1_v)} | P@5px: {np.mean(good_v):.1%}")
    axes[1].axis('off')
    
    # Plot Constrained Target Points
    axes[2].imshow(rgb1)
    good_c = errs_c < 5.0
    axes[2].scatter(m1_c[good_c, 0], m1_c[good_c, 1], c='g', s=20, alpha=0.8, marker='*', label='< 5px Error')
    if np.sum(~good_c) > 0:
        axes[2].scatter(m1_c[~good_c, 0], m1_c[~good_c, 1], c='r', s=20, alpha=0.8, marker='*', label='> 5px Error')
    axes[2].set_title(f"Our Model (CONSTRAINED EPIPOLAR)\\nTotal: {len(m1_c)} | P@5px: {np.mean(good_c):.1%}")
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.show()

print("Setup Complete.")
"""))

cells.append(nbf.v4.new_markdown_cell("## Comparison 1: Images 0 -> 20"))
cells.append(nbf.v4.new_code_cell("plot_benchmark_triplet(0, 20)"))

cells.append(nbf.v4.new_markdown_cell("## Comparison 2: Images 40 -> 60"))
cells.append(nbf.v4.new_code_cell("plot_benchmark_triplet(40, 60)"))

nb.cells.extend(cells)

with open('Benchmark_Visual_Analysis.ipynb', 'w') as f:
    nbf.write(nb, f)

print("Created Benchmark Notebook")
