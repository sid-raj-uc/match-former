from wxbs_benchmark.dataset import WxBSDataset

# By setting download=True, the package will fetch the dataset
# and store it in a local folder called 'WxBS' (by default).
dataset = WxBSDataset('WxBS_data_folder', download=True)

print(f"Successfully downloaded {len(dataset)} image pairs!")

# Let's look at the first pair
first_pair = dataset[0]
img1 = first_pair['img1']
img2 = first_pair['img2']
# ground_truth_matrix = first_pair['homography'] # The mathematical match

print("First pair loaded successfully!")