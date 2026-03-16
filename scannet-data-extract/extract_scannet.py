#!/usr/bin/env python3
"""
Download and extract ScanNet .sens files locally.

Usage:
    # Test parser on already-extracted scene (verify correctness):
    python extract_scannet.py --test

    # Download + extract specific scenes:
    python extract_scannet.py --scenes scene0001_00 scene0002_00

    # Just extract an already-downloaded .sens:
    python extract_scannet.py --sens /path/to/scene.sens --out /path/to/out_dir
"""
import struct, zlib, argparse, os, sys, ssl, shutil
import cv2
import numpy as np
from tqdm import tqdm

ssl._create_default_https_context = ssl._create_unverified_context

DATA_DIR   = os.path.join(os.path.dirname(__file__), 'data', 'scans')
SCANNET_PY = os.path.join(os.path.dirname(__file__), 'scannet.py')

COMPRESSION_COLOR = {-1:'unknown', 0:'raw', 1:'png', 2:'jpeg'}
COMPRESSION_DEPTH = {-1:'unknown', 0:'raw_ushort', 1:'zlib_ushort', 2:'occi_ushort'}


class RGBDFrame:
    def load(self, f):
        self.camera_to_world  = np.frombuffer(f.read(64), dtype=np.float32).reshape(4, 4).copy()
        self.timestamp_color  = struct.unpack('Q', f.read(8))[0]
        self.timestamp_depth  = struct.unpack('Q', f.read(8))[0]
        self.color_size_bytes = struct.unpack('Q', f.read(8))[0]
        self.depth_size_bytes = struct.unpack('Q', f.read(8))[0]
        self.color_data       = f.read(self.color_size_bytes)
        self.depth_data       = f.read(self.depth_size_bytes)





class SensorData:
    """
    .sens v4 binary layout (verified via hex dump of real ScanNet files):

      version              (I,  4 bytes)
      scene_name_len       (Q,  8 bytes)
      scene_name           (scene_name_len bytes)
      color_intrinsics     (4×4 float32 = 64 bytes)   ← row-major
      extrinsic_1          (4×4 float32 = 64 bytes)   ← identity
      depth_intrinsics     (4×4 float32 = 64 bytes)   ← row-major
      extrinsic_2          (4×4 float32 = 64 bytes)   ← identity
      color_compression    (i,  4 bytes)
      depth_compression    (i,  4 bytes)
      color_width          (I,  4 bytes)
      color_height         (I,  4 bytes)
      depth_width          (I,  4 bytes)
      depth_height         (I,  4 bytes)
      depth_shift          (f,  4 bytes)
      num_frames           (Q,  8 bytes)   ← comes AFTER dimensions

    Per frame (NO per-frame intrinsics):
      camera_to_world      (4×4 float32 = 64 bytes)
      timestamp_color      (Q,  8 bytes)
      timestamp_depth      (Q,  8 bytes)
      color_size_bytes     (Q,  8 bytes)
      depth_size_bytes     (Q,  8 bytes)
      color_data           (color_size_bytes)
      depth_data           (depth_size_bytes)
    """
    def __init__(self, filename):
        with open(filename, 'rb') as f:
            version = struct.unpack('I', f.read(4))[0]
            assert version == 4, f'Unsupported .sens version {version}'

            strlen = struct.unpack('Q', f.read(8))[0]
            self.scene_name = f.read(strlen).decode('utf-8', errors='replace')

            # Four fixed 4×4 float32 matrices: color intrinsics, identity, depth intrinsics, identity
            self.color_intrinsics = np.frombuffer(f.read(64), dtype=np.float32).reshape(4, 4).copy()
            f.read(64)  # extrinsic_1 (identity)
            self.depth_intrinsics = np.frombuffer(f.read(64), dtype=np.float32).reshape(4, 4).copy()
            f.read(64)  # extrinsic_2 (identity)

            self.color_compression = COMPRESSION_COLOR[struct.unpack('i', f.read(4))[0]]
            self.depth_compression = COMPRESSION_DEPTH[struct.unpack('i', f.read(4))[0]]
            self.color_width       = struct.unpack('I', f.read(4))[0]
            self.color_height      = struct.unpack('I', f.read(4))[0]
            self.depth_width       = struct.unpack('I', f.read(4))[0]
            self.depth_height      = struct.unpack('I', f.read(4))[0]
            self.depth_shift       = struct.unpack('f', f.read(4))[0]
            self.num_frames        = struct.unpack('Q', f.read(8))[0]

            print(f'  scene    : {self.scene_name}')
            print(f'  frames   : {self.num_frames}')
            print(f'  color    : {self.color_width}×{self.color_height} ({self.color_compression})')
            print(f'  depth    : {self.depth_width}×{self.depth_height} ({self.depth_compression})')
            print(f'  depth_fx : {self.depth_intrinsics[0,0]:.2f}  cx={self.depth_intrinsics[0,2]:.2f}')
            print(f'  depth_fy : {self.depth_intrinsics[1,1]:.2f}  cy={self.depth_intrinsics[1,2]:.2f}')

            self.frames = []
            for _ in tqdm(range(self.num_frames), desc='  Reading frames', unit='fr'):
                frame = RGBDFrame()
                frame.load(f)
                self.frames.append(frame)

    def export(self, out_dir):
        color_dir     = os.path.join(out_dir, 'color')
        depth_dir     = os.path.join(out_dir, 'depth')
        pose_dir      = os.path.join(out_dir, 'pose')
        intrinsic_dir = os.path.join(out_dir, 'intrinsic')
        for d in [color_dir, depth_dir, pose_dir, intrinsic_dir]:
            os.makedirs(d, exist_ok=True)

        np.savetxt(os.path.join(intrinsic_dir, 'intrinsic_color.txt'), self.color_intrinsics)
        np.savetxt(os.path.join(intrinsic_dir, 'intrinsic_depth.txt'), self.depth_intrinsics)

        for i, frame in enumerate(tqdm(self.frames, desc='  Extracting', unit='fr')):
            with open(os.path.join(color_dir, f'{i}.jpg'), 'wb') as cf:
                cf.write(frame.color_data)

            raw = zlib.decompress(frame.depth_data) \
                  if self.depth_compression == 'zlib_ushort' else frame.depth_data
            depth = np.frombuffer(raw, dtype=np.uint16).reshape(self.depth_height, self.depth_width)
            cv2.imwrite(os.path.join(depth_dir, f'{i}.png'), depth)

            np.savetxt(os.path.join(pose_dir, f'{i}.txt'), frame.camera_to_world)


def download_and_extract(scene_id, scratch_dir=None, skip_download=False):
    """Download .sens for scene_id, extract to DATA_DIR/scene_id/exported/."""
    out_dir      = os.path.join(DATA_DIR, scene_id)
    exported_dir = os.path.join(out_dir, 'exported')
    color_dir    = os.path.join(exported_dir, 'color')

    if os.path.exists(color_dir) and len(os.listdir(color_dir)) > 100:
        print(f'[SKIP] {scene_id} already extracted ({len(os.listdir(color_dir))} frames)')
        return

    # Use scene's own dir as scratch if not specified
    scratch_dir = scratch_dir or out_dir
    os.makedirs(scratch_dir, exist_ok=True)
    sens_path = os.path.join(scratch_dir, f'{scene_id}.sens')

    if not skip_download:
        sys.path.insert(0, os.path.dirname(SCANNET_PY))
        from scannet import download_scan
        download_scan(scene_id, scratch_dir, file_types=['.sens'],
                      use_v1_sens=True, skip_existing=True)

    print(f'\n── Parsing {scene_id} ({os.path.getsize(sens_path)/1024/1024:.0f} MB) ──')
    sd = SensorData(sens_path)
    sd.export(exported_dir)
    # Delete .sens to free disk space (exported/ is all we need)
    os.remove(sens_path)
    print(f'  Exported {len(os.listdir(color_dir))} frames → {exported_dir}  (.sens deleted)')


def test_parser():
    """Verify parser on scene0000_00 by comparing intrinsics with already-extracted files."""
    sens_path  = os.path.join(DATA_DIR, 'scene0000_00', 'scene0000_00.sens')
    truth_path = os.path.join(DATA_DIR, 'scene0000_00', 'exported', 'intrinsic', 'intrinsic_depth.txt')

    assert os.path.exists(sens_path), f'Not found: {sens_path}'
    assert os.path.exists(truth_path), f'Not found: {truth_path}'

    print(f'Parsing {sens_path} ...')
    sd = SensorData(sens_path)

    truth = np.loadtxt(truth_path)
    print(f'\nParsed depth intrinsics:\n{sd.depth_intrinsics}')
    print(f'\nGround-truth intrinsic_depth.txt:\n{truth}')
    match = np.allclose(sd.depth_intrinsics, truth, atol=1e-3)
    print(f'\nMatch: {match}')
    if not match:
        print('MISMATCH — parser needs further adjustment')
    else:
        print('Parser verified ✓')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--test', action='store_true',
                        help='Test parser on scene0000_00 (already extracted)')
    parser.add_argument('--scenes', nargs='+',
                        help='Scene IDs to download and extract')
    parser.add_argument('--sens', help='Path to a .sens file to extract directly')
    parser.add_argument('--out',  help='Output directory (used with --sens)')
    args = parser.parse_args()

    if args.test:
        test_parser()
    elif args.sens:
        assert args.out, '--out required with --sens'
        sd = SensorData(args.sens)
        sd.export(args.out)
    elif args.scenes:
        for scene_id in args.scenes:
            download_and_extract(scene_id)
    else:
        parser.print_help()
