import cv2
import numpy as np
import os

data_dir = 'data/scannet/test/scene0000_00'

color_img = cv2.imread(os.path.join(data_dir, 'color', '300.jpg'))
depth_img = cv2.imread(os.path.join(data_dir, 'depth', '300.png'), cv2.IMREAD_ANYDEPTH)

print('Color shape:', color_img.shape)
print('Depth shape:', depth_img.shape)

K_color = np.loadtxt(os.path.join(data_dir, 'intrinsic', 'intrinsic_color.txt'))[:3, :3]
K_depth = np.loadtxt(os.path.join(data_dir, 'intrinsic', 'intrinsic_depth.txt'))[:3, :3]

print('K_color:\n', K_color)
print('K_depth:\n', K_depth)

extrinsic_dir = os.path.join(data_dir, 'intrinsic')
if os.path.exists(os.path.join(extrinsic_dir, 'extrinsic_color.txt')):
    print('extrinsic_color:\n', np.loadtxt(os.path.join(extrinsic_dir, 'extrinsic_color.txt')))
if os.path.exists(os.path.join(extrinsic_dir, 'extrinsic_depth.txt')):
    print('extrinsic_depth:\n', np.loadtxt(os.path.join(extrinsic_dir, 'extrinsic_depth.txt')))
