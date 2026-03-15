import torch
import torch.nn.functional as F
import cv2
import numpy as np
import glob
from tqdm import tqdm

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


# --- BENCHMARK PIPELINE ---
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

def get_image_tensor(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (640, 480))
    img_tensor = torch.from_numpy(img).float() / 255.0
    return img_tensor.unsqueeze(0).unsqueeze(0)

# Evaluate Sampson distance
def compute_epipolar_errors(mkpts0, mkpts1, F):
    # mkpts0, mkpts1 are shape (N, 2)
    if len(mkpts0) == 0: return np.array([])
    p0 = np.concatenate([mkpts0, np.ones((len(mkpts0), 1))], axis=1)
    p1 = np.concatenate([mkpts1, np.ones((len(mkpts1), 1))], axis=1)
    
    l_prime = p0 @ F.T 
    num = np.abs(np.sum(l_prime * p1, axis=1))
    denom = np.sqrt(l_prime[:, 0]**2 + l_prime[:, 1]**2)
    return num / (denom + 1e-8)

def evaluate_pair(img0_idx, img1_idx):
    path0 = all_imgs[img0_idx]
    path1 = all_imgs[img1_idx]
    
    T0 = get_pose_for_image(path0, poses)
    T1 = get_pose_for_image(path1, poses)
    if T0 is None or T1 is None: return None
    
    F_mat = compute_fundamental_matrix(T0, T1, K, K)
    
    img0 = get_image_tensor(path0)
    img1 = get_image_tensor(path1)
    
    # 1. RUN VANILLA
    input_data = {'image0': img0, 'image1': img1}
    model.matcher.coarse_matching.epipolar_F = None
    with torch.no_grad():
        model.matcher(input_data)
        mkpts0_vanilla = input_data['mkpts0_f'].cpu().numpy()
        mkpts1_vanilla = input_data['mkpts1_f'].cpu().numpy()
        
    # 2. RUN CONSTRAINED
    input_data_constrained = {'image0': img0, 'image1': img1}
    model.matcher.coarse_matching.epipolar_F = F_mat
    with torch.no_grad():
        model.matcher(input_data_constrained)
        mkpts0_const = input_data_constrained['mkpts0_f'].cpu().numpy()
        mkpts1_const = input_data_constrained['mkpts1_f'].cpu().numpy()
        
    # Eval
    errs_v = compute_epipolar_errors(mkpts0_vanilla, mkpts1_vanilla, F_mat)
    errs_c = compute_epipolar_errors(mkpts0_const, mkpts1_const, F_mat)
    
    return {
        'v_total': len(errs_v),
        'v_p3': np.mean(errs_v < 3.0) if len(errs_v)>0 else 0,
        'v_p5': np.mean(errs_v < 5.0) if len(errs_v)>0 else 0,
        'c_total': len(errs_c),
        'c_p3': np.mean(errs_c < 3.0) if len(errs_c)>0 else 0,
        'c_p5': np.mean(errs_c < 5.0) if len(errs_c)>0 else 0,
    }

print("Running Benchmark...")
results = []
# Pick 10 image pairs separated by a gap (e.g., 20 frames apart so there's baseline)
for i in tqdm(range(0, 200, 20)):
    res = evaluate_pair(i, i+20)
    if res is not None:
        results.append(res)

print("\n--- RESULTS ---")
v_p3 = np.mean([r['v_p3'] for r in results])
v_p5 = np.mean([r['v_p5'] for r in results])
v_tot = np.mean([r['v_total'] for r in results])

c_p3 = np.mean([r['c_p3'] for r in results])
c_p5 = np.mean([r['c_p5'] for r in results])
c_tot = np.mean([r['c_total'] for r in results])

print(f"VANILLA:     P@3px: {v_p3:.2%} | P@5px: {v_p5:.2%} | Avg Total Matches: {v_tot:.1f}")
print(f"CONSTRAINED: P@3px: {c_p3:.2%} | P@5px: {c_p5:.2%} | Avg Total Matches: {c_tot:.1f}")
