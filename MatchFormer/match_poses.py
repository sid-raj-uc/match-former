import numpy as np
import os
import glob
from scipy.spatial.transform import Rotation as R

def read_trajectory(filename):
    poses = {}
    with open(filename, 'r') as f:
        for line in f:
            if line.startswith('#'):
                continue
            parts = line.strip().split()
            if len(parts) < 8:
                continue
            ts = float(parts[0])
            poses[ts] = {
                't': np.array([float(parts[1]), float(parts[2]), float(parts[3])]),
                'q': np.array([float(parts[4]), float(parts[5]), float(parts[6]), float(parts[7])])
            }
    return poses

def get_pose_for_image(img_path, poses):
    # Extract timestamp from filename (e.g., 1305031125.479424.png -> 1305031125.479424)
    basename = os.path.basename(img_path)
    ts_str = os.path.splitext(basename)[0]
    try:
        img_ts = float(ts_str)
    except ValueError:
        return None
        
    # Find closest timestamp in poses
    closest_ts = None
    min_diff = float('inf')
    
    for ts in poses.keys():
        diff = abs(ts - img_ts)
        if diff < min_diff:
            min_diff = diff
            closest_ts = ts
            
    # Assuming the difference should be extremely small (e.g. < 0.05 seconds)
    if min_diff > 0.05:
         print(f"Warning: Closest pose for {img_ts} is {min_diff}s away.")
         
    pose_data = poses[closest_ts]
    rot = R.from_quat(pose_data['q']).as_matrix()
    T = np.eye(4)
    T[:3, :3] = rot
    T[:3, 3] = pose_data['t']
    return T

if __name__ == "__main__":
    gt_file = '../tum_rgb_dataset/groundtruth.txt'
    rgb_dir = '../tum_rgb_dataset/rgb/*.png'
    
    poses = read_trajectory(gt_file)
    all_imgs = glob.glob(rgb_dir)
    all_imgs.sort() # Ensure deterministic order
    
    if len(all_imgs) >= 2:
        img1_path = all_imgs[0]  # Let's take the first two images as an example
        img2_path = all_imgs[10] 
        
        T1 = get_pose_for_image(img1_path, poses)
        T2 = get_pose_for_image(img2_path, poses)
        
        print(f"Pose for {os.path.basename(img1_path)}:\n{T1}")
        print(f"Pose for {os.path.basename(img2_path)}:\n{T2}")
