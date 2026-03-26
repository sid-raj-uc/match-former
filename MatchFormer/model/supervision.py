import torch
import numpy as np


@torch.no_grad()
def compute_supervision(data, config):
    """
    Populate data dict with GT coarse correspondence indices.

    Expects in data:
        'depth0'    [B, H, W]  depth in meters
        'K'         [B, 3, 3]  camera intrinsics
        'T0'        [B, 4, 4]  camera-to-world pose
        'T1'        [B, 4, 4]  camera-to-world pose
        'hw0_c'     (H_c, W_c) coarse feature grid size
        'hw0_i'     (H_i, W_i) input image size

    Adds to data:
        'spv_b_ids' [M]  batch indices of valid GT matches
        'spv_i_ids' [M]  flat coarse grid indices in Image 0
        'spv_j_ids' [M]  flat coarse grid indices in Image 1
    """
    device = data['image0'].device
    B = data['image0'].shape[0]

    H_img, W_img = data['hw0_i']
    H_c, W_c = data['hw0_c']
    stride = H_img // H_c  # typically 8

    K = data['K'].to(device)         # [B, 3, 3]
    T0 = data['T0'].to(device)       # [B, 4, 4]
    T1 = data['T1'].to(device)       # [B, 4, 4]
    depth0 = data['depth0'].to(device)  # [B, H, W]

    all_b_ids, all_i_ids, all_j_ids = [], [], []

    # Build coarse grid pixel coordinates (center of each stride block)
    ys = torch.arange(H_c, device=device).float() * stride + stride / 2  # [H_c]
    xs = torch.arange(W_c, device=device).float() * stride + stride / 2  # [W_c]
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing='ij')  # [H_c, W_c]
    grid_y_flat = grid_y.reshape(-1)  # [H_c*W_c]
    grid_x_flat = grid_x.reshape(-1)  # [H_c*W_c]
    N = H_c * W_c

    for b in range(B):
        fx, fy = K[b, 0, 0], K[b, 1, 1]
        cx, cy = K[b, 0, 2], K[b, 1, 2]

        # Sample depth at coarse grid positions (nearest neighbor)
        yi = grid_y_flat.long().clamp(0, H_img - 1)
        xi = grid_x_flat.long().clamp(0, W_img - 1)
        z = depth0[b, yi, xi]  # [N]

        # Filter: valid depth range
        valid = (z > 0.1) & (z < 10.0)
        if valid.sum() == 0:
            continue

        # Unproject to 3D in Camera 0 space
        xv = grid_x_flat[valid]
        yv = grid_y_flat[valid]
        zv = z[valid]

        X_c0 = (xv - cx) * zv / fx
        Y_c0 = (yv - cy) * zv / fy
        pts_c0 = torch.stack([X_c0, Y_c0, zv, torch.ones_like(zv)], dim=1)  # [M, 4]

        # Relative transform: Camera 0 -> World -> Camera 1
        T_12 = torch.linalg.inv(T1[b].float()) @ T0[b].float()  # [4, 4]

        # Transform points to Camera 1
        pts_c1 = (T_12 @ pts_c0.T).T  # [M, 4]

        # Filter: in front of Camera 1
        in_front = pts_c1[:, 2] > 0
        if in_front.sum() == 0:
            continue
        pts_c1 = pts_c1[in_front]
        valid_indices = torch.where(valid)[0][in_front]

        # Project to Image 1 pixels
        u1 = (pts_c1[:, 0] * fx / pts_c1[:, 2]) + cx
        v1 = (pts_c1[:, 1] * fy / pts_c1[:, 2]) + cy

        # Convert to coarse grid
        j_x = ((u1 - stride / 2) / stride).round().long()
        j_y = ((v1 - stride / 2) / stride).round().long()

        # Filter: within coarse grid bounds
        in_bounds = (j_x >= 0) & (j_x < W_c) & (j_y >= 0) & (j_y < H_c)
        if in_bounds.sum() == 0:
            continue

        i_ids = valid_indices[in_bounds]          # flat index in Image 0 coarse grid
        j_ids = j_y[in_bounds] * W_c + j_x[in_bounds]  # flat index in Image 1 coarse grid

        all_b_ids.append(torch.full((len(i_ids),), b, dtype=torch.long, device=device))
        all_i_ids.append(i_ids)
        all_j_ids.append(j_ids)

    if len(all_b_ids) == 0:
        # No valid GT matches in the batch — use empty tensors
        data.update({
            'spv_b_ids': torch.zeros(0, dtype=torch.long, device=device),
            'spv_i_ids': torch.zeros(0, dtype=torch.long, device=device),
            'spv_j_ids': torch.zeros(0, dtype=torch.long, device=device),
        })
    else:
        data.update({
            'spv_b_ids': torch.cat(all_b_ids),
            'spv_i_ids': torch.cat(all_i_ids),
            'spv_j_ids': torch.cat(all_j_ids),
        })
