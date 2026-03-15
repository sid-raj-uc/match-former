import numpy as np
import cv2
import os

from gt_epipolar import compute_fundamental_matrix

# Load Intrinsics
K = np.loadtxt('../data/scans/scene0000_00/exported/intrinsic/intrinsic_depth.txt')[:3, :3]

# Load Poses
T1 = np.loadtxt('../data/scans/scene0000_00/exported/pose/300.txt')
T2 = np.loadtxt('../data/scans/scene0000_00/exported/pose/320.txt')

print("T1 (Cam 1 to World):")
print(T1)

# In gt_epipolar.py:
# T_12 = np.linalg.inv(pose1) @ pose2
# This assumes poses are T_WC (Camera to World). 
# So T_12 = World_to_Cam1 @ Cam2_to_World = Cam2_to_Cam1

print("\nMethod 1 (Current):")
F_current = compute_fundamental_matrix(T1, T2, K, K)
print("F Matrix:")
print(F_current)

print("\nMethod 2 (Inverted Poses -> T_CW):")
# If ScanNet poses are actually T_CW (World to Camera), then 
# T_12 (Cam2 to Cam1) = CamW_to_Cam1 @ inv(CamW_to_Cam2) = pose1 @ np.linalg.inv(pose2)
F_inverted = compute_fundamental_matrix(np.linalg.inv(T1), np.linalg.inv(T2), K, K)
print("F Matrix (Inverted):")
print(F_inverted)

# Let's verify with an actual 3D point projection tests
pt1 = np.array([320, 240])

depth1 = cv2.imread('../data/scans/scene0000_00/exported/depth/300.png', cv2.IMREAD_ANYDEPTH)
z = depth1[240, 320] / 1000.0

cx, cy = K[0, 2], K[1, 2]
fx, fy = K[0, 0], K[1, 1]
x_c1 = (pt1[0] - cx) * z / fx
y_c1 = (pt1[1] - cy) * z / fy
p_c1 = np.array([x_c1, y_c1, z, 1.0])

print(f"\n3D point in C1: {p_c1}")

# Project using assumed T_WC
p_world_method1 = T1 @ p_c1
p_c2_method1 = np.linalg.inv(T2) @ p_world_method1
print(f"Projected C2 (Method 1): {p_c2_method1}")

# Project using assumed T_CW
p_world_method2 = np.linalg.inv(T1) @ p_c1
p_c2_method2 = T2 @ p_world_method2
print(f"Projected C2 (Method 2): {p_c2_method2}")

