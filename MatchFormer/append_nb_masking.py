import json
import os

NOTEBOOK_PATH = '/Users/siddharthraj/classes/cv/cv_final/MatchFormer/WxBS_Visual_Analysis.ipynb'

with open(NOTEBOOK_PATH, 'r') as f:
    nb = json.load(f)

new_cells = [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Epipolar Masked Predictions\n",
    "Here we enforce the Epipolar Geometry constraint mathematically onto the MatchFormer output matrix. We multiply the raw $N \\times N$ confidence matrix by a binary mask (thresholded by epipolar distance), zeroing out physically impossible predictions before sorting for the top 10 matches."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "def apply_epipolar_mask_to_conf_matrix(F, conf_matrix, H_img, W_img, stride, threshold=20.0):\n",
    "    H_c, W_c = H_img // stride, W_img // stride\n",
    "    N = H_c * W_c\n",
    "    \n",
    "    # Generate dense coarse coordinates\n",
    "    y, x = torch.meshgrid(torch.arange(H_c), torch.arange(W_c), indexing='ij')\n",
    "    coarse_coords = torch.stack([x, y], dim=-1).float().to(device)\n",
    "    coarse_coords = coarse_coords * stride + stride // 2\n",
    "    pts = coarse_coords.view(N, 2)\n",
    "    \n",
    "    pts_h = torch.cat([pts, torch.ones_like(pts[:, :1])], dim=-1)\n",
    "    F_t = torch.from_numpy(F).float().to(device)\n",
    "    \n",
    "    # Epipolar lines in image 2 for all points in image 1\n",
    "    l2 = F_t @ pts_h.T  # (3, N)\n",
    "    l2_norm = torch.sqrt(l2[0]**2 + l2[1]**2 + 1e-8)\n",
    "    dot_1_to_2 = pts_h @ l2 # (N, N) where (N_img2, N_img1)\n",
    "    dist_1to2 = torch.abs(dot_1_to_2) / l2_norm.unsqueeze(0)\n",
    "    \n",
    "    # Epipolar lines in image 1 for all points in image 2\n",
    "    l1 = F_t.T @ pts_h.T # (3, N)\n",
    "    l1_norm = torch.sqrt(l1[0]**2 + l1[1]**2 + 1e-8)\n",
    "    dot_2_to_1 = pts_h @ l1 # (N, N) where (N_img1, N_img2)\n",
    "    dist_2to1 = torch.abs(dot_2_to_1) / l1_norm.unsqueeze(0)\n",
    "    \n",
    "    # Symmetric dist: (N_img1, N_img2) to match conf_matrix\n",
    "    dist_sym = (dist_1to2.T + dist_2to1) / 2.0\n",
    "    \n",
    "    mask = (dist_sym < threshold).float()\n",
    "    return conf_matrix * mask\n",
    "\n",
    "def visualize_masked_vs_unmasked(category, pair_name, top_n=10, mask_threshold=20.0):\n",
    "    res = load_and_resize_pair(category, pair_name)\n",
    "    if res is None:\n",
    "        print(f\"Skipping {category}/{pair_name}...\")\n",
    "        return\n",
    "    img1, img2, gray1, gray2, pts1, pts2 = res\n",
    "\n",
    "    F, _ = cv2.findFundamentalMat(pts1, pts2, cv2.FM_8POINT)\n",
    "    if F is None or F.shape != (3, 3):\n",
    "        return\n",
    "\n",
    "    t0 = torch.from_numpy(gray1).unsqueeze(0).unsqueeze(0).to(device)\n",
    "    t1 = torch.from_numpy(gray2).unsqueeze(0).unsqueeze(0).to(device)\n",
    "    data = {'image0': t0, 'image1': t1}\n",
    "    with torch.no_grad():\n",
    "        model.matcher(data)\n",
    "\n",
    "    mkpts0 = data['mkpts0_f'].cpu().numpy()\n",
    "    mkpts1 = data['mkpts1_f'].cpu().numpy()\n",
    "    mconf_raw = data['mconf'].cpu().numpy()\n",
    "    \n",
    "    # Retrieve original NxN conf_matrix to apply masking\n",
    "    conf_matrix_raw = data['conf_matrix'][0] # shape (N, N)\n",
    "    \n",
    "    # Get base Top N indices directly from mconf for the unmasked set\n",
    "    sort_idx_raw = np.argsort(mconf_raw)[::-1][:min(top_n, len(mconf_raw))]\n",
    "    pts0_raw = mkpts0[sort_idx_raw]\n",
    "    pts1_raw = mkpts1[sort_idx_raw]\n",
    "    \n",
    "    # Compute Masked Top N\n",
    "    conf_matrix_masked = apply_epipolar_mask_to_conf_matrix(F, conf_matrix_raw, H_IMG, W_IMG, STRIDE, mask_threshold)\n",
    "    \n",
    "    # We need to replicate how match selection runs in PL_LoFTR to get the new matches, or simply argmax:\n",
    "    # For simplicity of exact 1-to-1 extraction over the matrix:\n",
    "    conf_masked = conf_matrix_masked.cpu().numpy()\n",
    "    flat_masked_idx = np.argsort(conf_masked.flatten())[::-1][:top_n]\n",
    "    \n",
    "    H_C, W_C = H_IMG // STRIDE, W_IMG // STRIDE\n",
    "    N = H_C * W_C\n",
    "    pts0_masked = []\n",
    "    pts1_masked = []\n",
    "    confs_masked = []\n",
    "    \n",
    "    for flat_i in flat_masked_idx:\n",
    "        conf_val = conf_masked.flatten()[flat_i]\n",
    "        idx0 = flat_i // N\n",
    "        idx1 = flat_i % N\n",
    "        pt0 = coarse_to_pixel(idx0)\n",
    "        pt1 = coarse_to_pixel(idx1)\n",
    "        pts0_masked.append(pt0)\n",
    "        pts1_masked.append(pt1)\n",
    "        confs_masked.append(conf_val)\n",
    "        \n",
    "    pts0_masked = np.array(pts0_masked)\n",
    "    pts1_masked = np.array(pts1_masked)\n",
    "    \n",
    "    def draw_points_error(ax, p0_arr, p1_arr, title):\n",
    "        h1, w1 = img1.shape[:2]\n",
    "        h2, w2 = img2.shape[:2]\n",
    "        vis_img = np.zeros((max(h1, h2), w1 + w2, 3), dtype=np.uint8)\n",
    "        vis_img[:h1, :w1] = img1\n",
    "        vis_img[:h2, w1:w1+w2] = img2\n",
    "        ax.imshow(vis_img)\n",
    "        \n",
    "        errors = []\n",
    "        colors = plt.cm.jet(np.linspace(0, 1, top_n))\n",
    "        for i in range(len(p0_arr)):\n",
    "            x1, y1 = p0_arr[i]\n",
    "            x2, y2 = p1_arr[i]\n",
    "            c = colors[i]\n",
    "            err = get_epipolar_distance(F, p0_arr[i], p1_arr[i])\n",
    "            errors.append(err)\n",
    "            ax.plot([x1, x2 + w1], [y1, y2], color=c, linewidth=2, marker='o', markersize=6)\n",
    "            \n",
    "        mean_err = np.mean(errors) if len(errors) > 0 else 0\n",
    "        ax.set_title(f\"{title}\\nMean Err: {mean_err:.2f}px\", fontsize=16)\n",
    "        ax.axis('off')\n",
    "        \n",
    "    fig, axes = plt.subplots(2, 1, figsize=(18, 16))\n",
    "    draw_points_error(axes[0], pts0_raw, pts1_raw, \"Vanilla MatchFormer (Unmasked)\")\n",
    "    draw_points_error(axes[1], pts0_masked, pts1_masked, f\"Epipolar Masked (Threshold={mask_threshold}px)\")\n",
    "    plt.suptitle(f\"Epipolar Overlap Analysis | {category}/{pair_name}\", fontsize=22)\n",
    "    plt.tight_layout(rect=[0, 0.03, 1, 0.95])\n",
    "    plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "print(\"Visualizing Masked vs Unmasked Top 10 Predictions:\")\n",
    "np.random.seed(999)\n",
    "for _ in range(2):\n",
    "    cat = np.random.choice(CATEGORIES)\n",
    "    cat_dir = os.path.join(DATA_DIR, cat)\n",
    "    if not os.path.exists(cat_dir): continue\n",
    "    pairs = [d for d in os.listdir(cat_dir) if os.path.isdir(os.path.join(cat_dir, d))]\n",
    "    if not pairs: continue\n",
    "    pair = np.random.choice(pairs)\n",
    "    \n",
    "    visualize_masked_vs_unmasked(cat, pair, top_n=10, mask_threshold=10.0)"
   ]
  }
]

# Ensure we cleanly append as lists of strings
for cell in new_cells:
    if cell['cell_type'] == 'code':
        source_str = cell['source'][0]
        # properly array-ify
        lines = [l + '\n' for l in source_str.split('\n')]
        if lines[-1] == '\n':
            lines = lines[:-1]
        else:
            lines[-1] = lines[-1].rstrip('\n')
        cell['source'] = lines

nb['cells'].extend(new_cells)

with open(NOTEBOOK_PATH, 'w') as f:
    json.dump(nb, f, indent=1)

print("Epipolar Masking blocks added!")
