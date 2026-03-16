"""
Simple ScanNet Dataset for fine-tuning MatchFormer.
Loads image pairs directly from the exported ScanNet directory — no NPZ manifests needed.
Each item yields image tensors, depth, intrinsics, poses, and pair names.
"""

import os
import glob
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


class ScanNetSimpleDataset(Dataset):
    """
    Loads consecutive image pairs from a ScanNet exported scene directory.
    Directory structure:
        data_dir/
            color/      *.jpg
            depth/      *.png  (uint16, mm)
            pose/       *.txt  (4x4 float, camera-to-world in OpenGL coords)
            intrinsic/  intrinsic_depth.txt  (4x4)

    Args:
        data_dir (str): path to exported scene directory
        frame_gap (int): gap between paired frames (default: 20)
        img_size (tuple): (W, H) to resize images to (default: (640, 480))
        max_pairs (int): max number of pairs to load (None = all)
        skip_invalid_depth (bool): skip pairs with invalid depth coverage
    """

    def __init__(self, data_dir, frame_gap=20, img_size=(640, 480),
                 max_pairs=None, skip_invalid_depth=True):
        self.data_dir = data_dir
        self.frame_gap = frame_gap
        self.img_size = img_size  # (W, H)
        self.skip_invalid_depth = skip_invalid_depth

        # Load intrinsics (just 1 file — fast)
        K_raw = np.loadtxt(os.path.join(data_dir, 'intrinsic', 'intrinsic_depth.txt'))
        self.K = K_raw[:3, :3].astype(np.float32)

        # Collect all frames sorted by index
        color_paths = sorted(
            glob.glob(os.path.join(data_dir, 'color', '*.jpg')),
            key=lambda x: int(os.path.basename(x).split('.')[0])
        )

        # Build valid pairs — store PATHS only, no np.loadtxt here
        self.pairs = []
        for i in range(len(color_paths) - frame_gap):
            idx0 = os.path.basename(color_paths[i]).split('.')[0]
            idx1 = os.path.basename(color_paths[i + frame_gap]).split('.')[0]

            pose0_path = os.path.join(data_dir, 'pose', f'{idx0}.txt')
            pose1_path = os.path.join(data_dir, 'pose', f'{idx1}.txt')
            depth0_path = os.path.join(data_dir, 'depth', f'{idx0}.png')

            # Only check file existence here (fast) — poses loaded lazily
            if not os.path.exists(pose0_path) or not os.path.exists(pose1_path):
                continue
            if not os.path.exists(depth0_path):
                continue

            self.pairs.append({
                'img0_path': color_paths[i],
                'img1_path': color_paths[i + frame_gap],
                'depth0_path': depth0_path,
                'pose0_path': pose0_path,
                'pose1_path': pose1_path,
                'idx0': idx0,
                'idx1': idx1,
            })

            if max_pairs and len(self.pairs) >= max_pairs:
                break

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        pair = self.pairs[idx]
        W, H = self.img_size

        # Load poses lazily (only when a batch actually needs this pair)
        T0 = np.loadtxt(pair['pose0_path']).astype(np.float32)
        T1 = np.loadtxt(pair['pose1_path']).astype(np.float32)

        # Skip invalid poses at getitem time
        if not np.isfinite(T0).all() or not np.isfinite(T1).all():
            # Return the next pair as fallback
            return self.__getitem__((idx + 1) % len(self.pairs))

        # Load and resize images
        img0 = cv2.imread(pair['img0_path'], cv2.IMREAD_GRAYSCALE)
        img1 = cv2.imread(pair['img1_path'], cv2.IMREAD_GRAYSCALE)
        img0 = cv2.resize(img0, (W, H)).astype(np.float32) / 255.0
        img1 = cv2.resize(img1, (W, H)).astype(np.float32) / 255.0

        # Load depth
        depth0_raw = cv2.imread(pair['depth0_path'], cv2.IMREAD_UNCHANGED)
        if depth0_raw is None:
            depth0 = np.zeros((H, W), dtype=np.float32)
        else:
            depth0 = cv2.resize(depth0_raw, (W, H), interpolation=cv2.INTER_NEAREST)
            depth0 = depth0.astype(np.float32) / 1000.0  # mm -> meters

        return {
            'image0': torch.from_numpy(img0).unsqueeze(0),   # [1, H, W]
            'image1': torch.from_numpy(img1).unsqueeze(0),   # [1, H, W]
            'depth0': torch.from_numpy(depth0),               # [H, W]
            'K': torch.from_numpy(self.K),                    # [3, 3]
            'T0': torch.from_numpy(T0),                       # [4, 4]
            'T1': torch.from_numpy(T1),                       # [4, 4]
            'pair_names': (pair['idx0'], pair['idx1']),
        }
