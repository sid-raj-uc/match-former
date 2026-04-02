import numpy as np
import os
import cv2
import torch

# TUM dataset typical intrinsics for the fr1 sequences
# K = [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]
# focal length fx = 517.3, fy = 516.5
# principal point cx = 318.6, cy = 255.3
# Ref: https://cvg.cit.tum.de/data/datasets/rgbd-dataset/file_formats
K = np.array([
    [517.3, 0.0, 318.6],
    [0.0, 516.5, 255.3],
    [0.0, 0.0, 1.0]
])

def compute_fundamental_matrix(pose1, pose2, K1, K2):
    """
    pose1, pose2: 4x4 homogenous transformation matrices from camera to world (T_wc)
    Then T_12 (Camera 1 to Camera 2) = T_w_c2 * T_c1_w = inv(pose2) @ pose1
    """
    T_12 = np.linalg.inv(pose2) @ pose1
    R = T_12[:3, :3]
    t = T_12[:3, 3]
    
    # Skew-symmetric matrix of t
    t_x = np.array([
        [0, -t[2], t[1]],
        [t[2], 0, -t[0]],
        [-t[1], t[0], 0]
    ])
    
    # Essential Matrix E = t_x * R
    E = t_x @ R
    
    # Fundamental Matrix F = K1^{-T} * E * K2^{-1}
    K1_inv = np.linalg.inv(K1)
    K2_inv = np.linalg.inv(K2)
    F = K1_inv.T @ E @ K2_inv
    
    # Normalize F
    F = F / F[2, 2]
    return F

H_IMG, W_IMG = 480, 640


def epipolar_distance_matrix(F_mat, H0, W0, H1, W1, device='cpu'):
    """Compute pairwise epipolar distance matrix between coarse grids.

    Returns [1, L, S] tensor of pixel distances from each img1 cell to the
    epipolar line of each img0 cell.
    """
    y0, x0 = torch.meshgrid(torch.arange(H0), torch.arange(W0), indexing='ij')
    x0_img = (x0.float() / W0) * W_IMG
    y0_img = (y0.float() / H0) * H_IMG
    y1, x1 = torch.meshgrid(torch.arange(H1), torch.arange(W1), indexing='ij')
    x1_img = (x1.float() / W1) * W_IMG
    y1_img = (y1.float() / H1) * H_IMG

    p0 = torch.stack([x0_img.flatten(), y0_img.flatten(),
                       torch.ones_like(x0_img.flatten())], dim=1).to(device)
    p1 = torch.stack([x1_img.flatten(), y1_img.flatten(),
                       torch.ones_like(x1_img.flatten())], dim=1).to(device)
    F_t = torch.tensor(F_mat, dtype=torch.float32, device=device)

    l_prime = p0 @ F_t.T                                   # [L, 3]
    num = torch.abs(l_prime @ p1.T)                         # [L, S]
    denom = torch.sqrt(l_prime[:, 0]**2 + l_prime[:, 1]**2).unsqueeze(1)
    distances = num / (denom + 1e-8)                        # [L, S]
    return distances.unsqueeze(0)                            # [1, L, S]


def get_epipolar_mask(F_mat, H0, W0, H1, W1, tau=10.0,
                      mode='laplacian', device='cpu'):
    """Soft epipolar mask with selectable decay.

    Args:
        mode: 'laplacian' — exp(-d / tau)   (current default)
              'gaussian'  — exp(-d² / (2 tau²))
              'hard'      — 1 if d < tau else 0

    Returns [1, L, S] mask.
    """
    distances = epipolar_distance_matrix(
        F_mat, H0, W0, H1, W1, device=device).squeeze(0)

    if mode == 'laplacian':
        mask = torch.exp(-distances / tau)
    elif mode == 'gaussian':
        mask = torch.exp(-distances**2 / (2 * tau**2))
    elif mode == 'hard':
        mask = (distances < tau).float()
    else:
        raise ValueError(f"Unknown mode '{mode}', use laplacian/gaussian/hard")

    return mask.unsqueeze(0)


def read_trajectory(filename):
    """
    Reads TUM trajectory file (groundtruth.txt)
    Format: timestamp tx ty tz qx qy qz qw
    """
    from scipy.spatial.transform import Rotation as R
    poses = {}
    with open(filename, 'r') as f:
        for line in f:
            if line.startswith('#'):
                continue
            parts = line.strip().split()
            if len(parts) < 8:
                continue
            ts = float(parts[0])
            t = np.array([float(parts[1]), float(parts[2]), float(parts[3])])
            q = np.array([float(parts[4]), float(parts[5]), float(parts[6]), float(parts[7])]) # x, y, z, w
            
            rot = R.from_quat(q).as_matrix()
            T = np.eye(4)
            T[:3, :3] = rot
            T[:3, 3] = t
            poses[ts] = T
    return poses

if __name__ == "__main__":
    gt_file = '../tum_rgb_dataset/groundtruth.txt'
    if os.path.exists(gt_file):
        print("Found groundtruth file.")
        poses = read_trajectory(gt_file)
        
        # In a real pipeline, we'd match the RGB image timestamps to these poses using nearest neighbor search.
        # But for now, we're just creating the utility.
        if len(poses) >= 2:
            ts1, ts2 = list(poses.keys())[0], list(poses.keys())[10] # just an example pair
            F = compute_fundamental_matrix(poses[ts1], poses[ts2], K, K)
            print("Computed F matrix shape:", F.shape)
            print(F)
        
    else:
        print(f"groundtruth file not found at {gt_file}")
