import json

NOTEBOOK_PATH = '/Users/siddharthraj/classes/cv/cv_final/MatchFormer/WxBS_Visual_Analysis.ipynb'
with open(NOTEBOOK_PATH, 'r') as f:
    nb = json.load(f)

new_cell = {
 "cell_type": "code",
 "execution_count": None,
 "metadata": {},
 "outputs": [],
 "source": [
  "print(\"Visualizing Masked vs Unmasked Top 10 Predictions:\")\n",
  "import os\n",
  "CATEGORIES = [d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d))]\n",
  "np.random.seed(999)\n",
  "for _ in range(2):\n",
  "    cat = np.random.choice(CATEGORIES)\n",
  "    cat_dir = os.path.join(DATA_DIR, cat)\n",
  "    if not os.path.exists(cat_dir): continue\n",
  "    pairs = [d for d in os.listdir(cat_dir) if os.path.isdir(os.path.join(cat_dir, d))]\n",
  "    if not pairs: continue\n",
  "    pair = np.random.choice(pairs)\n",
  "    \n",
  "    visualize_masked_vs_unmasked(cat, pair, top_n=10)"
 ]
}

nb['cells'].append(new_cell)

with open(NOTEBOOK_PATH, 'w') as f:
    json.dump(nb, f, indent=1)
