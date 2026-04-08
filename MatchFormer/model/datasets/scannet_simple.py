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
        frame_gap (int): gap between paired frames (default: 20).
            Used as fixed gap when random_gap_range is None.
        img_size (tuple): (W, H) to resize images to (default: (640, 480))
        max_pairs (int): cap total pairs (None = all)
        random_gap_range (tuple | None): (min_gap, max_gap) — sample a random
            gap uniformly from [min_gap, max_gap] per pair at each __getitem__.
            If None, uses fixed frame_gap. Pairs are indexed by source frame;
            the target frame is sampled on the fly.
        scenes (list[str] | None): list of scene names to include (e.g. ['scene0000_00', 'scene0001_00']).
            If None, uses all scenes found.
        split (str): 'train', 'test', or 'all'. Per-scene split based on frame index order.
        split_ratio (float): fraction of frames per scene used for training (default: 0.9).
    """

    def __init__(self, data_dir, frame_gap=20, img_size=(640, 480), max_pairs=None,
                 random_gap_range=None, scenes=None, split='all', split_ratio=0.9,
                 split_mode='sequential', split_seed=42):
        self.frame_gap = frame_gap
        self.img_size  = img_size
        self.random_gap_range = random_gap_range

        scene_dirs = self._resolve_scene_dirs(data_dir)

        # Filter to requested scenes
        if scenes is not None:
            scene_dirs = [d for d in scene_dirs
                          if any(s in d for s in scenes)]

        print(f'  Scenes found: {len(scene_dirs)} (split={split}, ratio={split_ratio}, mode={split_mode})')
        self.split = split
        self.split_ratio = split_ratio
        self.split_mode = split_mode
        self.split_seed = split_seed

        if random_gap_range is not None:
            self.frames = []
            max_gap = random_gap_range[1]
            for scene_dir in scene_dirs:
                scene_frames = self._build_frames(scene_dir, max_gap)
                scene_name = os.path.basename(os.path.dirname(scene_dir))
                print(f'    {scene_name}: {len(scene_frames)} source frames (gap range {random_gap_range})')
                self.frames.extend(scene_frames)
                if max_pairs and len(self.frames) >= max_pairs:
                    self.frames = self.frames[:max_pairs]
                    break
            self.pairs = None
            print(f'  Total source frames: {len(self.frames)}')
        else:
            self.frames = None
            self.pairs = []
            for scene_dir in scene_dirs:
                scene_pairs = self._build_pairs(scene_dir)
                scene_name  = os.path.basename(os.path.dirname(scene_dir))
                print(f'    {scene_name}: {len(scene_pairs)} pairs')
                self.pairs.extend(scene_pairs)
                if max_pairs and len(self.pairs) >= max_pairs:
                    self.pairs = self.pairs[:max_pairs]
                    break

            # Random split: shuffle all pairs, then take train/test portions
            if split_mode == 'random' and split != 'all':
                rng = np.random.RandomState(self.split_seed)
                indices = rng.permutation(len(self.pairs))
                split_idx = int(len(self.pairs) * self.split_ratio)
                if split == 'train':
                    self.pairs = [self.pairs[i] for i in indices[:split_idx]]
                else:  # test
                    self.pairs = [self.pairs[i] for i in indices[split_idx:]]

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

    def _get_split_range(self, n_frames):
        """Return (start, end) indices for the current split."""
        split_idx = int(n_frames * self.split_ratio)
        if self.split == 'train':
            return 0, split_idx
        elif self.split == 'test':
            return split_idx, n_frames
        else:
            return 0, n_frames

    def _build_frames(self, scene_dir, max_gap):
        """Build list of source frames that have valid pose/depth and at least
        max_gap frames ahead of them."""
        K_raw = np.loadtxt(os.path.join(scene_dir, 'intrinsic', 'intrinsic_depth.txt'))
        K = K_raw[:3, :3].astype(np.float32)

        color_paths = sorted(
            glob.glob(os.path.join(scene_dir, 'color', '*.jpg')),
            key=lambda x: int(os.path.basename(x).split('.')[0])
        )

        start, end = self._get_split_range(len(color_paths))

        frames = []
        for i in range(max(start, 0), min(end, len(color_paths) - max_gap)):
            idx = os.path.basename(color_paths[i]).split('.')[0]
            pose = os.path.join(scene_dir, 'pose', f'{idx}.txt')
            depth = os.path.join(scene_dir, 'depth', f'{idx}.png')
            if not os.path.exists(pose) or not os.path.exists(depth):
                continue
            frames.append({
                'scene_dir': scene_dir,
                'color_paths': color_paths,
                'src_index': i,
                'K': K,
            })
        return frames

    def _build_pairs(self, scene_dir):
        K_raw = np.loadtxt(os.path.join(scene_dir, 'intrinsic', 'intrinsic_depth.txt'))
        K = K_raw[:3, :3].astype(np.float32)

        color_paths = sorted(
            glob.glob(os.path.join(scene_dir, 'color', '*.jpg')),
            key=lambda x: int(os.path.basename(x).split('.')[0])
        )

        # Random split mode: use all frames, splitting happens after pair building
        if self.split_mode == 'random':
            start, end = 0, len(color_paths)
        else:
            start, end = self._get_split_range(len(color_paths))

        pairs = []
        for i in range(max(start, 0), min(end, len(color_paths)) - self.frame_gap):
            # Ensure the target frame is also within the split range
            j = i + self.frame_gap
            if j >= end:
                continue

            idx0 = os.path.basename(color_paths[i]).split('.')[0]
            idx1 = os.path.basename(color_paths[j]).split('.')[0]

            pose0  = os.path.join(scene_dir, 'pose',  f'{idx0}.txt')
            pose1  = os.path.join(scene_dir, 'pose',  f'{idx1}.txt')
            depth0 = os.path.join(scene_dir, 'depth', f'{idx0}.png')

            if not os.path.exists(pose0) or not os.path.exists(pose1):
                continue
            if not os.path.exists(depth0):
                continue

            # Skip pairs with invalid (nan/inf) poses
            T0 = np.loadtxt(pose0)
            T1 = np.loadtxt(pose1)
            if not np.isfinite(T0).all() or not np.isfinite(T1).all():
                continue

            pairs.append({
                'img0_path':   color_paths[i],
                'img1_path':   color_paths[j],
                'depth0_path': depth0,
                'pose0_path':  pose0,
                'pose1_path':  pose1,
                'idx0': idx0,
                'idx1': idx1,
                'K':    K,
            })
        return pairs

    def __len__(self):
        if self.frames is not None:
            return len(self.frames)
        return len(self.pairs)

    def _load_pair(self, img0_path, img1_path, depth0_path, pose0_path, pose1_path, K):
        W, H = self.img_size

        T0 = np.loadtxt(pose0_path).astype(np.float32)
        T1 = np.loadtxt(pose1_path).astype(np.float32)

        if not np.isfinite(T0).all() or not np.isfinite(T1).all():
            return None

        img0 = cv2.imread(img0_path, cv2.IMREAD_GRAYSCALE)
        img1 = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)
        img0 = cv2.resize(img0, (W, H)).astype(np.float32) / 255.0
        img1 = cv2.resize(img1, (W, H)).astype(np.float32) / 255.0

        depth0_raw = cv2.imread(depth0_path, cv2.IMREAD_UNCHANGED)
        if depth0_raw is None:
            depth0 = np.zeros((H, W), dtype=np.float32)
        else:
            depth0 = cv2.resize(depth0_raw, (W, H), interpolation=cv2.INTER_NEAREST)
            depth0 = depth0.astype(np.float32) / 1000.0  # mm → metres

        idx0 = os.path.basename(img0_path).split('.')[0]
        idx1 = os.path.basename(img1_path).split('.')[0]

        return {
            'image0':     torch.from_numpy(img0).unsqueeze(0),
            'image1':     torch.from_numpy(img1).unsqueeze(0),
            'depth0':     torch.from_numpy(depth0),
            'K':          torch.from_numpy(K),
            'T0':         torch.from_numpy(T0),
            'T1':         torch.from_numpy(T1),
            'pair_names': (idx0, idx1),
            'hw0_i':      (H, W),
            'hw0_c':      (H // 8, W // 8),
        }

    def __getitem__(self, idx):
        if self.frames is not None:
            return self._getitem_random_gap(idx)
        return self._getitem_fixed_gap(idx)

    def _getitem_random_gap(self, idx):
        frame = self.frames[idx]
        min_gap, max_gap = self.random_gap_range
        color_paths = frame['color_paths']
        src_i = frame['src_index']

        # Sample random gap, retry on invalid pose
        for _ in range(10):
            gap = np.random.randint(min_gap, max_gap + 1)
            tgt_i = src_i + gap
            if tgt_i >= len(color_paths):
                continue

            idx0 = os.path.basename(color_paths[src_i]).split('.')[0]
            idx1 = os.path.basename(color_paths[tgt_i]).split('.')[0]
            scene_dir = frame['scene_dir']

            pose0 = os.path.join(scene_dir, 'pose', f'{idx0}.txt')
            pose1 = os.path.join(scene_dir, 'pose', f'{idx1}.txt')
            depth0 = os.path.join(scene_dir, 'depth', f'{idx0}.png')

            if not os.path.exists(pose1):
                continue

            result = self._load_pair(color_paths[src_i], color_paths[tgt_i],
                                     depth0, pose0, pose1, frame['K'])
            if result is not None:
                return result

        # Fallback: use min_gap
        tgt_i = src_i + min_gap
        idx0 = os.path.basename(color_paths[src_i]).split('.')[0]
        idx1 = os.path.basename(color_paths[tgt_i]).split('.')[0]
        scene_dir = frame['scene_dir']
        return self._load_pair(
            color_paths[src_i], color_paths[tgt_i],
            os.path.join(scene_dir, 'depth', f'{idx0}.png'),
            os.path.join(scene_dir, 'pose', f'{idx0}.txt'),
            os.path.join(scene_dir, 'pose', f'{idx1}.txt'),
            frame['K'])

    def _getitem_fixed_gap(self, idx):
        pair = self.pairs[idx]
        result = self._load_pair(
            pair['img0_path'], pair['img1_path'], pair['depth0_path'],
            pair['pose0_path'], pair['pose1_path'], pair['K'])
        if result is None:
            return self._getitem_fixed_gap((idx + 1) % len(self.pairs))
        return result
