import matplotlib.pyplot as plt
import numpy as np
import cv2

def plot_epipolar_attention(img1, img2, query_pt, F, attention_maps, layer_names):
    """
    img1, img2: RGB images (H, W, 3)
    query_pt: (x, y) coordinates in img1
    F: 3x3 Fundamental matrix (maps point in img1 to line in img2)
    attention_maps: List of (H_feat, W_feat) attention heatmaps for the given query_pt
    layer_names: List of strings for plot titles
    """
    
    # 1. Epipolar line in img2: l' = F * p
    p = np.array([query_pt[0], query_pt[1], 1.0])
    l_prime = F @ p 
    
    # Line equation: a*x + b*y + c = 0 => y = -(a*x + c)/b
    a, b, c = l_prime
    
    num_layers = len(attention_maps)
    cols = 3
    rows = (num_layers + 1) // cols + 1
    
    fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))
    axes = axes.flatten()
    
    # Plot Image 1 with Query Point
    axes[0].imshow(img1)
    axes[0].plot(query_pt[0], query_pt[1], 'r*', markersize=15)
    axes[0].set_title("Source Image (Query Point)")
    axes[0].axis('off')
    
    h, w = img2.shape[:2]
    
    for i, (attn_map, name) in enumerate(zip(attention_maps, layer_names)):
        ax = axes[i + 1]
        
        # Resize attention map to original image resolution for overlay
        attn_resized = cv2.resize(attn_map, (w, h))
        
        # Normalize for visualization
        attn_resized = (attn_resized - attn_resized.min()) / (attn_resized.max() - attn_resized.min() + 1e-8)
        
        ax.imshow(img2)
        # Overlay heatmap
        ax.imshow(attn_resized, cmap='jet', alpha=0.5)
        
        # Draw Epipolar line
        x0, x1 = 0, w
        y0 = int(-(a*x0 + c) / b)
        y1 = int(-(a*x1 + c) / b)
        
        ax.plot([x0, x1], [y0, y1], 'w--', linewidth=2, label='Epipolar Line')
        
        ax.set_title(f"Target Image - {name}")
        ax.axis('off')

    # Hide extra subplots
    for j in range(i + 2, len(axes)):
        axes[j].axis('off')
        
    plt.tight_layout()
    plt.show()
