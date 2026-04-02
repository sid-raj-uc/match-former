import json
import os

NOTEBOOK_PATH = '/Users/siddharthraj/classes/cv/cv_final/MatchFormer/WxBS_Visual_Analysis.ipynb'
with open(NOTEBOOK_PATH, 'r') as f:
    nb = json.load(f)

for cell in nb['cells']:
    if cell['cell_type'] == 'code' and 'def visualize_masked_vs_unmasked' in ''.join(cell['source']):
        source = """\
def get_epipolar_distance(F, p1, p2):
    # p1, p2 are (x, y)
    h1 = np.array([p1[0], p1[1], 1.0])
    h2 = np.array([p2[0], p2[1], 1.0])
    
    l2 = F @ h1
    d2 = np.abs(np.dot(h2, l2)) / np.sqrt(l2[0]**2 + l2[1]**2 + 1e-8)
    
    l1 = F.T @ h2
    d1 = np.abs(np.dot(h1, l1)) / np.sqrt(l1[0]**2 + l1[1]**2 + 1e-8)
    
    return (d1 + d2) / 2.0

def visualize_masked_vs_unmasked(category, pair_name, top_n=10):
    res = load_and_resize_pair(category, pair_name)
    if res is None:
        print(f"Skipping {category}/{pair_name}...")
        return
    img1, img2, gray1, gray2, pts1, pts2 = res

    F, _ = cv2.findFundamentalMat(pts1, pts2, cv2.FM_8POINT)
    if F is None:
        print(f"Failed to find F for {category}/{pair_name}")
        return
    if F.shape != (3, 3):
        F = F[:3, :]

    t0 = torch.from_numpy(gray1).unsqueeze(0).unsqueeze(0).to(device)
    t1 = torch.from_numpy(gray2).unsqueeze(0).unsqueeze(0).to(device)
    data = {'image0': t0, 'image1': t1}
    
    # 1. Vanilla (Unmasked) Execution
    model.matcher.coarse_matching.epipolar_F = None
    with torch.no_grad():
        model.matcher(data)

    mconf_raw = data['mconf'].cpu().numpy()
    sort_idx_raw = np.argsort(mconf_raw)[::-1][:min(top_n, len(mconf_raw))]
    pts0_raw = data['mkpts0_f'].cpu().numpy()[sort_idx_raw]
    pts1_raw = data['mkpts1_f'].cpu().numpy()[sort_idx_raw]
    
    # 2. Epipolar Masked Execution
    # Convert F to torch tensor on the correct device if MatchFormer expects a tensor natively! 
    # Let's pass F exactly as it is (numpy array) since Quad_Visual_Analysis passes `F` directly as calculated by findFundamentalMat.
    model.matcher.coarse_matching.epipolar_F = F
    with torch.no_grad():
        model.matcher(data)
        
    mconf_masked = data['mconf'].cpu().numpy()
    sort_idx_masked = np.argsort(mconf_masked)[::-1][:min(top_n, len(mconf_masked))]
    pts0_masked = data['mkpts0_f'].cpu().numpy()[sort_idx_masked]
    pts1_masked = data['mkpts1_f'].cpu().numpy()[sort_idx_masked]
    
    def draw_points_error(ax, p0_arr, p1_arr, title):
        h1, w1 = img1.shape[:2]
        h2, w2 = img2.shape[:2]
        vis_img = np.zeros((max(h1, h2), w1 + w2, 3), dtype=np.uint8)
        vis_img[:h1, :w1] = img1
        vis_img[:h2, w1:w1+w2] = img2
        ax.imshow(vis_img)
        
        errors = []
        colors = plt.cm.jet(np.linspace(0, 1, len(p0_arr)))
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
    draw_points_error(axes[1], pts0_masked, pts1_masked, "Native Epipolar Enforced")
    plt.suptitle(f"Epipolar Overlap Analysis | {category}/{pair_name}", fontsize=22)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()
"""
        lines = [l + '\n' for l in source.split('\n')]
        if lines[-1] == '\n':
            lines = lines[:-1]
        else:
            lines[-1] = lines[-1].rstrip('\n')
        cell['source'] = lines

with open(NOTEBOOK_PATH, 'w') as f:
    json.dump(nb, f, indent=1)
