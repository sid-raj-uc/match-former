import numpy as np
import os
import cv2

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
    pose1, pose2: 4x4 homogenous transformation matrices from camera to world (or vice-versa)
    Let's assume they are Camera to World (T_wc)
    Then T_12 (Camera 2 to Camera 1) = T_c1_w * T_w_c2 = inv(pose1) * pose2
    """
    T_12 = np.linalg.inv(pose1) @ pose2
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
