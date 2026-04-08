import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

# Paths
base_dir = 'data/scannet_format/MH_05'
idx0, idx1 = 500, 520 # Frame gap of 20

img0_path = os.path.join(base_dir, 'color', f'{idx0}.jpg')
img1_path = os.path.join(base_dir, 'color', f'{idx1}.jpg')
pose0_path = os.path.join(base_dir, 'pose', f'{idx0}.txt')
pose1_path = os.path.join(base_dir, 'pose', f'{idx1}.txt')
K_path = os.path.join(base_dir, 'intrinsic', 'intrinsic_depth.txt')

img0 = cv2.imread(img0_path)
img1 = cv2.imread(img1_path)

K = np.loadtxt(K_path)
T0 = np.loadtxt(pose0_path)
T1 = np.loadtxt(pose1_path)

# Relative pose from 0 to 1
# T_WC gives camera to world. So T_CW = inv(T_WC)
# T_01 transforms points from frame 0 to frame 1: P1 = T_CW_1 @ T_WC_0 @ P0
T01 = np.linalg.inv(T1) @ T0
R = T01[:3, :3]
t = T01[:3, 3]

# Essential matrix
tx = np.array([
    [0, -t[2], t[1]],
    [t[2], 0, -t[0]],
    [-t[1], t[0], 0]
])
E = tx @ R

# Fundamental matrix
K_inv = np.linalg.inv(K)
F = K_inv.T @ E @ K_inv

# Get a point using SIFT to act as our pseudo-GT point
sift = cv2.SIFT_create()
kp0, des0 = sift.detectAndCompute(img0, None)
kp1, des1 = sift.detectAndCompute(img1, None)

bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
matches = bf.match(des0, des1)
matches = sorted(matches, key=lambda x: x.distance)

# Pick a good match
match = matches[5] # Picking 5th best to avoid edge cases
pt0 = np.array(kp0[match.queryIdx].pt)
pt1 = np.array(kp1[match.trainIdx].pt)

# Compute epipolar line in image 1 corresponding to pt0 in image 0
# Line equation: l = F @ [pt0.x, pt0.y, 1]^T
pt0_h = np.array([pt0[0], pt0[1], 1.0])
l1 = F @ pt0_h

# Draw the line on image 1
# l1[0]*x + l1[1]*y + l1[2] = 0
# y = (-l1[2] - l1[0]*x) / l1[1]

h, w = img1.shape[:2]
x0, w_x = 0, w

y0 = int((-l1[2] - l1[0]*x0) / l1[1])
y1 = int((-l1[2] - l1[0]*w_x) / l1[1])

# Draw line
img1_line = img1.copy()
cv2.line(img1_line, (x0, y0), (w_x, y1), (0, 255, 0), 2)
# Draw point
cv2.circle(img1_line, (int(pt1[0]), int(pt1[1])), 5, (0, 0, 255), -1)

# Draw point on img0
img0_pt = img0.copy()
cv2.circle(img0_pt, (int(pt0[0]), int(pt0[1])), 5, (0, 0, 255), -1)

# Plot
fig, axes = plt.subplots(1, 2, figsize=(15, 5))
axes[0].imshow(cv2.cvtColor(img0_pt, cv2.COLOR_BGR2RGB))
axes[0].set_title('Image 0 with Selected Point')
axes[0].axis('off')

axes[1].imshow(cv2.cvtColor(img1_line, cv2.COLOR_BGR2RGB))
axes[1].set_title('Image 1 with Epipolar Line and SIFT Match (Red)')
axes[1].axis('off')

plt.tight_layout()
artifact_path = '/Users/siddharthraj/.gemini/antigravity/brain/6623aa8e-2dd6-496f-802a-efac64538e56/artifacts/epipolar_vis.png'
os.makedirs(os.path.dirname(artifact_path), exist_ok=True)
plt.savefig(artifact_path)
print(f"Saved visualization to {artifact_path}")
