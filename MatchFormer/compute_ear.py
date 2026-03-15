import numpy as np
import math

def distance_point_to_line(x0, y0, a, b, c):
    """
    Distance from point (x0, y0) to line ax + by + c = 0
    d = |a*x0 + b*y0 + c| / sqrt(a^2 + b^2)
    """
    return abs(a*x0 + b*y0 + c) / math.sqrt(a**2 + b**2)

def compute_EAR(attention_map, a, b, c, k_pixels):
    """
    Computes Epipolar Attention Ratio (EAR).
    EAR = sum(attention within k pixels of line) / sum(all attention)
    
    attention_map: 2D numpy array of attention weights (H, W)
    a, b, c: coefficients of the epipolar line equation ax + by + c = 0
    k_pixels: threshold distance in pixels
    """
    H, W = attention_map.shape
    
    # Create coordinate grids
    y_coords, x_coords = np.mgrid[0:H, 0:W]
    
    # Compute distances for all points in grid
    # d = |a*x + b*y + c| / sqrt(a^2 + b^2)
    distances = np.abs(a*x_coords + b*y_coords + c) / math.sqrt(a**2 + b**2)
    
    # Create mask for points within k_pixels
    mask = distances <= k_pixels
    
    attention_sum_within_k = np.sum(attention_map[mask])
    total_attention = np.sum(attention_map)
    
    if total_attention == 0:
        return 0.0
        
    return attention_sum_within_k / total_attention

def random_baseline_EAR(H, W, a, b, c, k_pixels):
    """
    Approximates the EAR if attention was completely uniformly distributed.
    Since uniform attention assigns weight 1/(H*W) to every pixel, 
    the baseline EAR is just the ratio of pixels within the mask to the total pixels.
    """
    y_coords, x_coords = np.mgrid[0:H, 0:W]
    distances = np.abs(a*x_coords + b*y_coords + c) / math.sqrt(a**2 + b**2)
    mask = distances <= k_pixels
    pixels_in_mask = np.sum(mask)
    return pixels_in_mask / (H * W)
