"""
Visualize GT match projections to verify correctness.
Draws query points on image0 and their depth-projected GT locations on image1,
using both coordinate conventions (with and without T_cv2gl).
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import glob

H_img, W_img = 480, 640

def get_scene_dir(scene_name, base='../data/scans'):
    exported = os.path.join(base, scene_name, 'exported')
    if os.path.isdir(os.path.join(exported, 'color')):
        return exported
    return os.path.join(base, scene_name)


def project_point(pt, depth_path, T0, T1, K, use_cv2gl=False):
    """Project a point from image0 to image1 using depth + poses."""
    depth = cv2.imread(depth_path, cv2.IMREAD_ANYDEPTH)
    if depth is None:
        return None
    if depth.shape != (H_img, W_img):
        depth = cv2.resize(depth, (W_img, H_img), interpolation=cv2.INTER_NEAREST)

    ux, uy = int(round(pt[0])), int(round(pt[1]))
    z = depth[uy, ux] / 1000.0
    if z <= 0.1 or z > 10.0:
        return None

    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    # Unproject to 3D in camera 0 frame
    x_c = (pt[0] - cx) * z / fx
    y_c = (pt[1] - cy) * z / fy
    p_c0 = np.array([x_c, y_c, z, 1.0])

    # Transform to camera 1 frame
    if use_cv2gl:
        T_cv2gl = np.array([[1,0,0,0],[0,-1,0,0],[0,0,-1,0],[0,0,0,1]], dtype=np.float64)
        T_12 = T_cv2gl @ np.linalg.inv(T1) @ T0 @ T_cv2gl
    else:
        T_12 = np.linalg.inv(T1) @ T0

    p_c1 = T_12 @ p_c0
    if p_c1[2] <= 0:
        return None

    u2 = (p_c1[0] * fx / p_c1[2]) + cx
    v2 = (p_c1[1] * fy / p_c1[2]) + cy

    if 0 <= u2 < W_img and 0 <= v2 < H_img:
        return np.array([u2, v2])
    return None


def visualize_pair(scene_name, idx0, idx1, query_points, frame_gap=20):
    data_dir = get_scene_dir(scene_name)
    all_imgs = sorted(glob.glob(f'{data_dir}/color/*.jpg'),
                      key=lambda x: int(os.path.basename(x).split('.')[0]))
    K = np.loadtxt(f'{data_dir}/intrinsic/intrinsic_depth.txt')[:3, :3]

    img0_path = all_imgs[idx0]
    img1_path = all_imgs[idx1]
    idx0_num = os.path.basename(img0_path).split('.')[0]
    idx1_num = os.path.basename(img1_path).split('.')[0]

    T0 = np.loadtxt(os.path.join(data_dir, 'pose', f'{idx0_num}.txt'))
    T1 = np.loadtxt(os.path.join(data_dir, 'pose', f'{idx1_num}.txt'))

    img0 = cv2.cvtColor(cv2.resize(cv2.imread(img0_path), (W_img, H_img)), cv2.COLOR_BGR2RGB)
    img1 = cv2.cvtColor(cv2.resize(cv2.imread(img1_path), (W_img, H_img)), cv2.COLOR_BGR2RGB)
    depth_path = os.path.join(data_dir, 'depth', f'{idx0_num}.png')

    fig, axes = plt.subplots(2, 2, figsize=(20, 12))
    fig.suptitle(f'{scene_name} | Frames {idx0_num} → {idx1_num}', fontsize=14, fontweight='bold')

    # Top row: without T_cv2gl
    axes[0, 0].imshow(img0)
    axes[0, 0].set_title('Image 0 — Query Points', fontsize=12)
    axes[0, 0].axis('off')

    axes[0, 1].imshow(img1)
    axes[0, 1].set_title('Image 1 — GT Projection (WITHOUT T_cv2gl)', fontsize=12)
    axes[0, 1].axis('off')

    # Bottom row: with T_cv2gl
    axes[1, 0].imshow(img0)
    axes[1, 0].set_title('Image 0 — Query Points', fontsize=12)
    axes[1, 0].axis('off')

    axes[1, 1].imshow(img1)
    axes[1, 1].set_title('Image 1 — GT Projection (WITH T_cv2gl)', fontsize=12)
    axes[1, 1].axis('off')

    colors = ['red', 'lime', 'cyan', 'magenta', 'yellow', 'orange', 'white']

    for i, (ux, uy) in enumerate(query_points):
        c = colors[i % len(colors)]

        # Without T_cv2gl
        axes[0, 0].plot(ux, uy, '*', color=c, markersize=18, markeredgecolor='black', markeredgewidth=1)
        axes[0, 0].annotate(f'P{i}', (ux+8, uy-8), color=c, fontsize=11, fontweight='bold')

        gt_no_flip = project_point((ux, uy), depth_path, T0, T1, K, use_cv2gl=False)
        if gt_no_flip is not None:
            axes[0, 1].plot(gt_no_flip[0], gt_no_flip[1], '*', color=c, markersize=18,
                           markeredgecolor='black', markeredgewidth=1)
            axes[0, 1].annotate(f'P{i} ({gt_no_flip[0]:.0f},{gt_no_flip[1]:.0f})',
                               (gt_no_flip[0]+8, gt_no_flip[1]-8), color=c, fontsize=9, fontweight='bold')
        else:
            axes[0, 1].set_title(axes[0, 1].get_title() + f'\n(P{i} failed)', fontsize=10)

        # With T_cv2gl
        axes[1, 0].plot(ux, uy, '*', color=c, markersize=18, markeredgecolor='black', markeredgewidth=1)
        axes[1, 0].annotate(f'P{i}', (ux+8, uy-8), color=c, fontsize=11, fontweight='bold')

        gt_flip = project_point((ux, uy), depth_path, T0, T1, K, use_cv2gl=True)
        if gt_flip is not None:
            axes[1, 1].plot(gt_flip[0], gt_flip[1], '*', color=c, markersize=18,
                           markeredgecolor='black', markeredgewidth=1)
            axes[1, 1].annotate(f'P{i} ({gt_flip[0]:.0f},{gt_flip[1]:.0f})',
                               (gt_flip[0]+8, gt_flip[1]-8), color=c, fontsize=9, fontweight='bold')
        else:
            axes[1, 1].set_title(axes[1, 1].get_title() + f'\n(P{i} failed)', fontsize=10)

    plt.tight_layout()
    safe_scene = scene_name.replace('/', '_')
    out_path = f'gt_verify_{safe_scene}_{idx0_num}_{idx1_num}.png'
    plt.savefig(out_path, dpi=120, bbox_inches='tight')
    print(f'Saved: {out_path}')
    plt.close()


if __name__ == '__main__':
    np.random.seed(42)

    pairs = [
        ('scene0011_00', 100, 120),
        ('scene0012_00', 200, 220),
        ('scene0013_00', 50, 70),
    ]

    # 5 query points spread across the image
    query_points = [
        (160, 120),   # top-left area
        (480, 120),   # top-right area
        (320, 240),   # center
        (160, 360),   # bottom-left area
        (480, 360),   # bottom-right area
    ]

    for scene, i0, i1 in pairs:
        print(f'\nProcessing {scene} frames {i0} → {i1}...')
        visualize_pair(scene, i0, i1, query_points)

    print('\nDone. Check the saved PNG files.')
