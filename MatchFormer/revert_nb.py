import json
import os

NOTEBOOK_PATH = '/Users/siddharthraj/classes/cv/cv_final/MatchFormer/WxBS_Visual_Analysis.ipynb'
with open(NOTEBOOK_PATH, 'r') as f:
    nb = json.load(f)

source = """\
def apply_epipolar_mask_to_conf_matrix(F, conf_matrix, H_img, W_img, stride, threshold=20.0):
    H_c, W_c = H_img // stride, W_img // stride
    N = H_c * W_c
    
    # Generate dense coarse coordinates
    y, x = torch.meshgrid(torch.arange(H_c), torch.arange(W_c), indexing='ij')
    coarse_coords = torch.stack([x, y], dim=-1).float().to(device)
    coarse_coords = coarse_coords * stride + stride // 2
    pts = coarse_coords.view(N, 2)
    
    pts_h = torch.cat([pts, torch.ones_like(pts[:, :1])], dim=-1)
    F_t = torch.from_numpy(F).float().to(device)
    
    # Epipolar lines in image 2 for all points in image 1
    l2 = F_t @ pts_h.T  # (3, N)
    l2_norm = torch.sqrt(l2[0]**2 + l2[1]**2 + 1e-8)
    dot_1_to_2 = pts_h @ l2 # (N, N) where (N_img2, N_img1)
    dist_1to2 = torch.abs(dot_1_to_2) / l2_norm.unsqueeze(0)
    
    # Epipolar lines in image 1 for all points in image 2
    l1 = F_t.T @ pts_h.T # (3, N)
    l1_norm = torch.sqrt(l1[0]**2 + l1[1]**2 + 1e-8)
    dot_2_to_1 = pts_h @ l1 # (N, N) where (N_img1, N_img2)
    dist_2to1 = torch.abs(dot_2_to_1) / l1_norm.unsqueeze(0)
    
    # Symmetric dist: (N_img1, N_img2) to match conf_matrix
    dist_sym = (dist_1to2.T + dist_2to1) / 2.0
    
    mask = (dist_sym < threshold).float()
    return conf_matrix * mask

def get_epipolar_distance(F, p1, p2):
    # p1, p2 are (x, y)
    h1 = np.array([p1[0], p1[1], 1.0])
    h2 = np.array([p2[0], p2[1], 1.0])
    
    l2 = F @ h1
    d2 = np.abs(np.dot(h2, l2)) / np.sqrt(l2[0]**2 + l2[1]**2 + 1e-8)
    
    l1 = F.T @ h2
    d1 = np.abs(np.dot(h1, l1)) / np.sqrt(l1[0]**2 + l1[1]**2 + 1e-8)
    
    return (d1 + d2) / 2.0

def visualize_masked_vs_unmasked(category, pair_name, top_n=10, mask_threshold=20.0):
    res = load_and_resize_pair(category, pair_name)
    if res is None:
        print(f"Skipping {category}/{pair_name}...")
        return
    img1, img2, gray1, gray2, pts1, pts2 = res

    F, _ = cv2.findFundamentalMat(pts1, pts2, cv2.FM_8POINT)
    if F is None: return
    if F.shape != (3, 3): F = F[:3, :]

    t0 = torch.from_numpy(gray1).unsqueeze(0).unsqueeze(0).to(device)
    t1 = torch.from_numpy(gray2).unsqueeze(0).unsqueeze(0).to(device)
    data = {'image0': t0, 'image1': t1}
    with torch.no_grad():
        model.matcher(data)

    mkpts0 = data['mkpts0_f'].cpu().numpy()
    mkpts1 = data['mkpts1_f'].cpu().numpy()
    mconf_raw = data['mconf'].cpu().numpy()
    
    # Retrieve original NxN conf_matrix to apply masking
    conf_matrix_raw = data['conf_matrix'][0] # shape (N, N)
    
    # Get base Top N indices directly from mconf for the unmasked set
    sort_idx_raw = np.argsort(mconf_raw)[::-1][:min(top_n, len(mconf_raw))]
    pts0_raw = mkpts0[sort_idx_raw]
    pts1_raw = mkpts1[sort_idx_raw]
    
    # Compute Masked Top N
    conf_matrix_masked = apply_epipolar_mask_to_conf_matrix(F, conf_matrix_raw, H_IMG, W_IMG, STRIDE, mask_threshold)
    
    conf_masked = conf_matrix_masked.cpu().numpy()
    flat_masked_idx = np.argsort(conf_masked.flatten())[::-1][:top_n]
    
    H_C, W_C = H_IMG // STRIDE, W_IMG // STRIDE
    N = H_C * W_C
    pts0_masked = []
    pts1_masked = []
    
    for flat_i in flat_masked_idx:
        idx0 = flat_i // N
        idx1 = flat_i % N
        pt0 = coarse_to_pixel(idx0)
        pt1 = coarse_to_pixel(idx1)
        pts0_masked.append(pt0)
        pts1_masked.append(pt1)
        
    pts0_masked = np.array(pts0_masked)
    pts1_masked = np.array(pts1_masked)
    
    def draw_points_error(ax, p0_arr, p1_arr, title):
        h1, w1 = img1.shape[:2]
        h2, w2 = img2.shape[:2]
        vis_img = np.zeros((max(h1, h2), w1 + w2, 3), dtype=np.uint8)
        vis_img[:h1, :w1] = img1
        vis_img[:h2, w1:w1+w2] = img2
        ax.imshow(vis_img)
        
        errors = []
        colors = plt.cm.jet(np.linspace(0, 1, top_n))
        for i in range(len(p0_arr)):
            x1, y1 = p0_arr[i]
            x2, y2 = p1_arr[i]
            c = colors[i]
            err = get_epipolar_distance(F, p0_arr[i], p1_arr[i])
            errors.append(err)
            ax.plot([x1, x2 + w1], [y1, y2], color=c, linewidth=2, marker='o', markersize=6)
            
        mean_err = np.mean(errors) if len(errors) > 0 else 0
        ax.set_title(f"{title}\\nMean Err: {mean_err:.2f}px", fontsize=16)
        ax.axis('off')
        
    fig, axes = plt.subplots(2, 1, figsize=(18, 16))
    draw_points_error(axes[0], pts0_raw, pts1_raw, "Vanilla MatchFormer (Unmasked)")
    draw_points_error(axes[1], pts0_masked, pts1_masked, f"Epipolar Masked Output (Threshold={mask_threshold}px)")
    plt.suptitle(f"Epipolar Overlap Analysis | {category}/{pair_name}", fontsize=22)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()
"""

lines = [l + '\n' for l in source.split('\n')]
if lines[-1] == '\n':
    lines = lines[:-1]
else:
    lines[-1] = lines[-1].rstrip('\n')

# Find the cell to replace
for cell in nb['cells']:
    if cell['cell_type'] == 'code' and 'def visualize_masked_vs_unmasked' in ''.join(cell['source']):
        cell['source'] = lines
        
with open(NOTEBOOK_PATH, 'w') as f:
    json.dump(nb, f, indent=1)
