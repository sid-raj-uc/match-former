import nbformat as nbf

nb = nbf.v4.new_notebook()

nb.cells = [
    nbf.v4.new_markdown_cell("# WxBS Visual Analysis\nDrawing motivation from Quad_Visual_Analysis, this notebook computes the Fundamental matrix from ground truth correspondences and plots the epipolar line and GT match."),
    nbf.v4.new_code_cell("""\
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import glob

# Set the path correctly
DATA_DIR = '/Users/siddharthraj/classes/cv/cv_final/data/WxBS_data_folder/v1.1'
"""),
    nbf.v4.new_code_cell("""\
def load_wxbs_pair(category, pair_name):
    base_path = os.path.join(DATA_DIR, category, pair_name)
    img1_path = os.path.join(base_path, '01.png')
    img2_path = os.path.join(base_path, '02.png')
    corrs_path = os.path.join(base_path, 'corrs.txt')
    
    img1 = cv2.cvtColor(cv2.imread(img1_path), cv2.COLOR_BGR2RGB)
    img2 = cv2.cvtColor(cv2.imread(img2_path), cv2.COLOR_BGR2RGB)
    
    corrs = np.loadtxt(corrs_path)
    pts1 = corrs[:, :2]
    pts2 = corrs[:, 2:]
    
    return img1, img2, pts1, pts2
"""),
    nbf.v4.new_code_cell("""\
def visualize_wxbs_epipolar(category, pair_name, pt_idx=0):
    img1, img2, pts1, pts2 = load_wxbs_pair(category, pair_name)
    
    # Compute F using 8-point algorithm on all points
    F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_8POINT)
    
    # Pick the point
    x1, y1 = pts1[pt_idx]
    x2, y2 = pts2[pt_idx]
    
    # Epipolar line in img2: l' = F * p1
    p1 = np.array([x1, y1, 1.0])
    l_prime = F @ p1
    a, b, c = l_prime
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 7))
    
    # Image 1
    axes[0].imshow(img1)
    axes[0].plot(x1, y1, 'r*', markersize=15)
    axes[0].set_title(f"Image 1 (Query Point)\\n{category}/{pair_name}")
    axes[0].axis('off')
    
    # Image 2
    axes[1].imshow(img2)
    axes[1].plot(x2, y2, 'g*', markersize=15, label='GT Match')
    
    # Draw epipolar line
    h, w = img2.shape[:2]
    x0, x_end = 0, w
    
    if b != 0:
        y0 = int(-(a * x0 + c) / b)
        y_end = int(-(a * x_end + c) / b)
        axes[1].plot([x0, x_end], [y0, y_end], 'w--', linewidth=2, label='Epipolar Line')
    elif a != 0:
        x_val = int(-c / a)
        axes[1].plot([x_val, x_val], [0, h], 'w--', linewidth=2, label='Epipolar Line')
        
    axes[1].set_title("Image 2: Epipolar Line & GT Match")
    axes[1].legend()
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.show()
"""),
    nbf.v4.new_code_cell("""\
# Example usage: visualize a random point from WGALBS/bridge
visualize_wxbs_epipolar('WGALBS', 'bridge', pt_idx=5)
""")
]

with open('WxBS_Visual_Analysis.ipynb', 'w') as f:
    nbf.write(nb, f)
print("Notebook created.")
