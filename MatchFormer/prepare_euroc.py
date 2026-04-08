import os
import glob
import cv2
import yaml
import numpy as np
import argparse
from scipy.spatial.transform import Rotation

def parse_sensor_yaml(yaml_path):
    with open(yaml_path, 'r') as f:
        # Load raw text to circumvent standard yaml loading issues with PyYAML versions
        lines = f.readlines()
        
    intrinsics = None
    T_BS = np.eye(4)
    
    in_T_BS = False
    t_bs_data = []
    
    for line in lines:
        if 'intrinsics:' in line:
            # e.g., intrinsics: [458.654, 457.296, 367.215, 248.375]
            parts = line.split('[')[1].split(']')[0]
            intrinsics = [float(x) for x in parts.split(',')]
        if 'T_BS:' in line:
            in_T_BS = True
            continue
        if in_T_BS:
            if 'data:' in line:
                part = line.split('[')[1]
                if ']' in part:
                    part = part.split(']')[0]
                    in_T_BS = False
                t_bs_data.extend([float(x.strip()) for x in part.split(',') if x.strip()])
            elif ']' in line:
                part = line.split(']')[0]
                in_T_BS = False
                t_bs_data.extend([float(x.strip()) for x in part.split(',') if x.strip()])
            elif 'rows' not in line and 'cols' not in line:
                t_bs_data.extend([float(x.strip()) for x in line.split(',') if x.strip()])
                
    T_BS = np.array(t_bs_data).reshape(4, 4)
    return intrinsics, T_BS

def load_groundtruth(csv_path):
    gt_data = []
    with open(csv_path, 'r') as f:
        for i, line in enumerate(f):
            if i == 0 or line.startswith('#'): continue
            parts = line.strip().split(',')
            ts = int(parts[0])
            p_x, p_y, p_z = map(float, parts[1:4])
            q_w, q_x, q_y, q_z = map(float, parts[4:8])
            
            T_RS = np.eye(4)
            # scipy Rotation uses [x, y, z, w]
            T_RS[:3, :3] = Rotation.from_quat([q_x, q_y, q_z, q_w]).as_matrix()
            T_RS[:3, 3] = [p_x, p_y, p_z]
            
            gt_data.append((ts, T_RS))
            
    gt_data.sort(key=lambda x: x[0])
    ts_array = np.array([x[0] for x in gt_data])
    pos_array = [x[1] for x in gt_data]
    return ts_array, pos_array

def process_sequence(seq_path, out_dir):
    cam0_dir = os.path.join(seq_path, 'mav0', 'cam0', 'data')
    gt_csv = os.path.join(seq_path, 'mav0', 'state_groundtruth_estimate0', 'data.csv')
    sensor_yaml = os.path.join(seq_path, 'mav0', 'cam0', 'sensor.yaml')
    
    print(f"Processing {seq_path}...")
    intrinsics, T_BS = parse_sensor_yaml(sensor_yaml)
    print("Intrinsics [fx, fy, cx, cy]:", intrinsics)
    print("T_BS parsed.")
    
    gt_ts, gt_poses = load_groundtruth(gt_csv)
    print(f"Loaded {len(gt_ts)} ground truth states.")
    
    img_paths = sorted(glob.glob(os.path.join(cam0_dir, '*.png')))
    print(f"Found {len(img_paths)} images.")
    
    # Create required subdirs
    os.makedirs(os.path.join(out_dir, 'color'), exist_ok=True)
    os.makedirs(os.path.join(out_dir, 'depth'), exist_ok=True)
    os.makedirs(os.path.join(out_dir, 'pose'), exist_ok=True)
    os.makedirs(os.path.join(out_dir, 'intrinsic'), exist_ok=True)
    
    fx, fy, cx, cy = intrinsics
    K = np.array([[fx, 0, cx],
                  [0, fy, cy],
                  [0, 0, 1]])
    
    np.savetxt(os.path.join(out_dir, 'intrinsic', 'intrinsic_depth.txt'), K, fmt='%.6f')
    np.savetxt(os.path.join(out_dir, 'intrinsic', 'intrinsic_color.txt'), K, fmt='%.6f')
    
    # Process frames
    for i, img_path in enumerate(img_paths):
        ts = int(os.path.basename(img_path).split('.')[0])
        
        # Closest gt log
        idx = np.abs(gt_ts - ts).argmin()
        T_RS = gt_poses[idx]
        
        # T_WC = T_RS * T_BS
        T_WC = T_RS @ T_BS
        
        # Save pose
        np.savetxt(os.path.join(out_dir, 'pose', f"{i}.txt"), T_WC, fmt='%.6f')
        
        # Read image to save as .jpg
        img = cv2.imread(img_path)
        if img is None:
            continue
        h, w = img.shape[:2]
        
        cv2.imwrite(os.path.join(out_dir, 'color', f"{i}.jpg"), img)
        
        # Empty depth
        depth = np.zeros((h, w), dtype=np.uint16)
        cv2.imwrite(os.path.join(out_dir, 'depth', f"{i}.png"), depth)
        
        if (i+1) % 500 == 0:
            print(f"Processed {i+1}/{len(img_paths)} frames.")
            
    print(f"Finished writing to {out_dir}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--seq', type=str, required=True, help="Path to sequence like data/machine_hall/MH_01_easy")
    parser.add_argument('--out', type=str, required=True, help="Output sequence directory like data/scannet_format/MH_01")
    args = parser.parse_args()
    
    process_sequence(args.seq, args.out)
