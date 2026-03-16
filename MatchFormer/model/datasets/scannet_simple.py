"""
Simple ScanNet Dataset for fine-tuning MatchFormer.
Supports multiple scenes by passing a root dir or list of exported scene dirs.
"""

import os
import glob
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


class ScanNetSimpleDataset(Dataset):
    """
    Loads consecutive image pairs from one or more ScanNet exported scene directories.

    Args:
        data_dir (str | list[str]):
            - path to a single exported scene dir (has color/ inside), OR
            - root dir containing scene subdirs (e.g. scans/ with scene000X_00/exported/), OR
            - list of exported scene directories
        frame_gap (int): gap between paired frames (default: 20)
        img_size (tuple): (W, H) to resize images to (default: (640, 480))
        max_pairs (int): cap total pairs (None = all)
    """

    def __init__(self, data_dir, frame_gap=20, img_size=(640, 480), max_pairs=None):
        self.frame_gap = frame_gap
        self.img_size  = img_size

        scene_dirs = self._resolve_scene_dirs(data_dir)
        print(f'  Scenes found: {len(scene_dirs)}')

        self.pairs = []
        for scene_dir in scene_dirs:
            scene_pairs = self._build_pairs(scene_dir)
            scene_name  = os.path.basename(os.path.dirname(scene_dir))
            print(f'    {scene_name}: {len(scene_pairs)} pairs')
            self.pairs.extend(scene_pairs)
            if max_pairs and len(self.pairs) >= max_pairs:
                self.pairs = self.pairs[:max_pairs]
                break

        print(f'  Total pairs: {len(self.pairs)}')

    @staticmethod
    def _resolve_scene_dirs(data_dir):
        if isinstance(data_dir, (list, tuple)):
            return [d for d in data_dir if os.path.isdir(os.path.join(d, 'color'))]

        # Single exported dir
        if os.path.isdir(os.path.join(data_dir, 'color')):
            return [data_dir]

        # Root dir — look for scene subdirs
        candidates = []
        for entry in sorted(os.listdir(data_dir)):
            exported = os.path.join(data_dir, entry, 'exported')
            if os.path.isdir(os.path.join(exported, 'color')):
                candidates.append(exported)
            elif os.path.isdir(os.path.join(data_dir, entry, 'color')):
                candidates.append(os.path.join(data_dir, entry))
        return candidates

    def _build_pairs(self, scene_dir):
        K_raw = np.loadtxt(os.path.join(scene_dir, 'intrinsic', 'intrinsic_depth.txt'))
        K = K_raw[:3, :3].astype(np.float32)

        color_paths = sorted(
            glob.glob(os.path.join(scene_dir, 'color', '*.jpg')),
            key=lambda x: int(os.path.basename(x).split('.')[0])
        )

        pairs = []
        for i in range(len(color_paths) - self.frame_gap):
            idx0 = os.path.basename(color_paths[i]).split('.')[0]
            idx1 = os.path.basename(color_paths[i + self.frame_gap]).split('.')[0]

            pose0  = os.path.join(scene_dir, 'pose',  f'{idx0}.txt')
            pose1  = os.path.join(scene_dir, 'pose',  f'{idx1}.txt')
            depth0 = os.path.join(scene_dir, 'depth', f'{idx0}.png')

            if not os.path.exists(pose0) or not os.path.exists(pose1):
                continue
            if not os.path.exists(depth0):
                continue

            pairs.append({
                'img0_path':   color_paths[i],
                'img1_path':   color_paths[i + self.frame_gap],
                'depth0_path': depth0,
                'pose0_path':  pose0,
                'pose1_path':  pose1,
                'idx0': idx0,
                'idx1': idx1,
                'K':    K,
            })
        return pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        pair = self.pairs[idx]
        W, H = self.img_size

        T0 = np.loadtxt(pair['pose0_path']).astype(np.float32)
        T1 = np.loadtxt(pair['pose1_path']).astype(np.float32)

        if not np.isfinite(T0).all() or not np.isfinite(T1).all():
            return self.__getitem__((idx + 1) % len(self.pairs))

        img0 = cv2.imread(pair['img0_path'], cv2.IMREAD_GRAYSCALE)
        img1 = cv2.imread(pair['img1_path'], cv2.IMREAD_GRAYSCALE)
        img0 = cv2.resize(img0, (W, H)).astype(np.float32) / 255.0
        img1 = cv2.resize(img1, (W, H)).astype(np.float32) / 255.0

        depth0_raw = cv2.imread(pair['depth0_path'], cv2.IMREAD_UNCHANGED)
        if depth0_raw is None:
            depth0 = np.zeros((H, W), dtype=np.float32)
        else:
            depth0 = cv2.resize(depth0_raw, (W, H), interpolation=cv2.INTER_NEAREST)
            depth0 = depth0.astype(np.float32) / 1000.0  # mm → metres

        return {
            'image0':     torch.from_numpy(img0).unsqueeze(0),
            'image1':     torch.from_numpy(img1).unsqueeze(0),
            'depth0':     torch.from_numpy(depth0),
            'K':          torch.from_numpy(pair['K']),
            'T0':         torch.from_numpy(T0),
            'T1':         torch.from_numpy(T1),
            'pair_names': (pair['idx0'], pair['idx1']),
            'hw0_i':      (H, W),
            'hw0_c':      (H // 8, W // 8),
        }
