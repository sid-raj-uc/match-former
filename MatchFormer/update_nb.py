import json
import os

path = '/Users/siddharthraj/classes/cv/cv_final/MatchFormer/WxBS_Visual_Analysis.ipynb'
with open(path, 'r') as f:
    nb = json.load(f)

# Find the cell with global matches
for cell in nb['cells']:
    if cell['cell_type'] == 'code' and 'def visualize_top_global_matches' in ''.join(cell['source']):
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

def visualize_top_global_matches(category, pair_name, top_n=10):
    res = load_and_resize_pair(category, pair_name)
    if res is None:
        print(f"Skipping {category}/{pair_name}...")
        return
    img1, img2, gray1, gray2, pts1, pts2 = res

    # Compute Fundamental Matrix F
    F, _ = cv2.findFundamentalMat(pts1, pts2, cv2.FM_8POINT)
    if F is None or F.shape != (3, 3):
        print(f"Fundamental matrix estimation failed for {category}/{pair_name}")
        return

    t0 = torch.from_numpy(gray1).unsqueeze(0).unsqueeze(0).to(device)
    t1 = torch.from_numpy(gray2).unsqueeze(0).unsqueeze(0).to(device)
    data = {'image0': t0, 'image1': t1}
    with torch.no_grad():
        model.matcher(data)

    mkpts0 = data['mkpts0_f'].cpu().numpy()
    mkpts1 = data['mkpts1_f'].cpu().numpy()
    mconf = data['mconf'].cpu().numpy()

    # Sort by confidence
    sort_idx = np.argsort(mconf)[::-1]
    top_idx = sort_idx[:min(top_n, len(mconf))]

    top_pts0 = mkpts0[top_idx]
    top_pts1 = mkpts1[top_idx]
    top_confs = mconf[top_idx]

    # Calculate Epipolar Pixel Error for each match
    errors = []
    for i in range(len(top_pts0)):
        err = get_epipolar_distance(F, top_pts0[i], top_pts1[i])
        errors.append(err)
    
    mean_error = np.mean(errors) if len(errors) > 0 else 0

    # Create a concatenated image for side-by-side match drawing
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    vis_img = np.zeros((max(h1, h2), w1 + w2, 3), dtype=np.uint8)
    vis_img[:h1, :w1] = img1
    vis_img[:h2, w1:w1+w2] = img2

    fig, ax = plt.subplots(figsize=(18, 9))
    ax.imshow(vis_img)

    colors = plt.cm.jet(np.linspace(0, 1, top_n))
    for i in range(len(top_pts0)):
        x1, y1 = top_pts0[i]
        x2, y2 = top_pts1[i]
        c = colors[i]
        err = errors[i]
        ax.plot([x1, x2 + w1], [y1, y2], color=c, linewidth=2, marker='o', markersize=6,
                label=f'Rank {i+1} (conf: {top_confs[i]:.3f}) | Err: {err:.1f}px')

    ax.axis('off')
    ax.set_title(f"Top {top_n} Matches | {category}/{pair_name} | Mean Err: {mean_error:.2f}px", fontsize=18)
    
    # Draw legend outside the plot box
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
    plt.tight_layout()
    plt.show()
"""
        # Split source into list of strings with newlines for JSON
        # This accurately mirrors Jupyter's internal format
        lines = [line + '\n' for line in source.split('\n')]
        lines[-1] = lines[-1].rstrip('\n') # remove trailing newline from last line
        cell['source'] = lines
        break

with open(path, 'w') as f:
    json.dump(nb, f, indent=1)
