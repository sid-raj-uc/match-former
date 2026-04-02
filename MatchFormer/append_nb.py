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
    "## Global Top 10 Matches\n",
    "Let's visualize the globally highest confidence matches predicted by MatchFormer across the whole image pair."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "def visualize_top_global_matches(category, pair_name, top_n=10):\n",
    "    res = load_and_resize_pair(category, pair_name)\n",
    "    if res is None:\n",
    "        print(f\"Skipping {category}/{pair_name}...\")\n",
    "        return\n",
    "    img1, img2, gray1, gray2, pts1, pts2 = res\n",
    "\n",
    "    t0 = torch.from_numpy(gray1).unsqueeze(0).unsqueeze(0).to(device)\n",
    "    t1 = torch.from_numpy(gray2).unsqueeze(0).unsqueeze(0).to(device)\n",
    "    data = {'image0': t0, 'image1': t1}\n",
    "    with torch.no_grad():\n",
    "        model.matcher(data)\n",
    "\n",
    "    mkpts0 = data['mkpts0_f'].cpu().numpy()\n",
    "    mkpts1 = data['mkpts1_f'].cpu().numpy()\n",
    "    mconf = data['mconf'].cpu().numpy()\n",
    "\n",
    "    # Sort by confidence\n",
    "    sort_idx = np.argsort(mconf)[::-1]\n",
    "    top_idx = sort_idx[:min(top_n, len(mconf))]\n",
    "\n",
    "    top_pts0 = mkpts0[top_idx]\n",
    "    top_pts1 = mkpts1[top_idx]\n",
    "    top_confs = mconf[top_idx]\n",
    "\n",
    "    # Create a concatenated image for side-by-side match drawing\n",
    "    h1, w1 = img1.shape[:2]\n",
    "    h2, w2 = img2.shape[:2]\n",
    "    vis_img = np.zeros((max(h1, h2), w1 + w2, 3), dtype=np.uint8)\n",
    "    vis_img[:h1, :w1] = img1\n",
    "    vis_img[:h2, w1:w1+w2] = img2\n",
    "\n",
    "    fig, ax = plt.subplots(figsize=(18, 9))\n",
    "    ax.imshow(vis_img)\n",
    "\n",
    "    colors = plt.cm.jet(np.linspace(0, 1, top_n))\n",
    "    for i in range(len(top_pts0)):\n",
    "        x1, y1 = top_pts0[i]\n",
    "        x2, y2 = top_pts1[i]\n",
    "        c = colors[i]\n",
    "        ax.plot([x1, x2 + w1], [y1, y2], color=c, linewidth=2, marker='o', markersize=6,\n",
    "                label=f'Rank {i+1} (conf={top_confs[i]:.3f})')\n",
    "\n",
    "    ax.axis('off')\n",
    "    ax.set_title(f\"Top {top_n} Matches | {category}/{pair_name}\", fontsize=18)\n",
    "    \n",
    "    # Draw legend outside the plot box\n",
    "    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left')\n",
    "    plt.tight_layout()\n",
    "    plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "print(\"Visualizing Top 10 Matches for 3 random pairs:\")\n",
    "np.random.seed(1337) # different seed for variety\n",
    "for _ in range(3):\n",
    "    cat = np.random.choice(CATEGORIES)\n",
    "    cat_dir = os.path.join(DATA_DIR, cat)\n",
    "    if not os.path.exists(cat_dir): continue\n",
    "    pairs = [d for d in os.listdir(cat_dir) if os.path.isdir(os.path.join(cat_dir, d))]\n",
    "    if not pairs: continue\n",
    "    pair = np.random.choice(pairs)\n",
    "    \n",
    "    visualize_top_global_matches(cat, pair, top_n=10)"
   ]
  }
]

nb['cells'].extend(new_cells)

with open(NOTEBOOK_PATH, 'w') as f:
    json.dump(nb, f, indent=1)

print("Notebook updated successfully with Top-10 Global Matches logic!")
