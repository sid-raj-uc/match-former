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

    # Compute fundamental matrices and epipolar mask for self-supervised loss
    F_list = []
    epi_masks = []
    epi_thresh = getattr(config, 'EPI_THRESH', 2.0)  # pixels

    # Grid positions in pixel coords for Image 0 and Image 1
    y0g, x0g = torch.meshgrid(
        torch.arange(H_c, device=device).float(),
        torch.arange(W_c, device=device).float(), indexing='ij')
    pos0_x = (x0g + 0.5) / W_c * W_img  # [H_c, W_c]
    pos0_y = (y0g + 0.5) / H_c * H_img
    # Homogeneous: [L, 3]
    p0h = torch.stack([pos0_x.flatten(), pos0_y.flatten(),
                       torch.ones(H_c * W_c, device=device)], dim=1)

    y1g, x1g = torch.meshgrid(
        torch.arange(H_c, device=device).float(),
        torch.arange(W_c, device=device).float(), indexing='ij')
    pos1_x = (x1g + 0.5) / W_c * W_img
    pos1_y = (y1g + 0.5) / H_c * H_img
    p1h = torch.stack([pos1_x.flatten(), pos1_y.flatten(),
                       torch.ones(H_c * W_c, device=device)], dim=1)  # [S, 3]

    for b in range(B):
        T0_np = T0[b].cpu().float().numpy()
        T1_np = T1[b].cpu().float().numpy()
        K_np = K[b].cpu().float().numpy()
        T_12 = np.linalg.inv(T1_np) @ T0_np
        R = T_12[:3, :3]
        t = T_12[:3, 3]
        t_x = np.array([
            [0, -t[2], t[1]],
            [t[2], 0, -t[0]],
            [-t[1], t[0], 0]
        ])
        E = t_x @ R
        K_inv = np.linalg.inv(K_np)
        F_mat = K_inv.T @ E @ K_inv
        if abs(F_mat[2, 2]) > 1e-10:
            F_mat = F_mat / F_mat[2, 2]
        F_list.append(F_mat)

        # Epipolar mask: for each point i in img0, mark cells in img1
        # within epi_thresh pixels of the epipolar line l = F @ p0
        F_t = torch.tensor(F_mat, dtype=torch.float32, device=device)
        lines = p0h @ F_t.T  # [L, 3] — epipolar lines in image 1
        # Point-to-line distance: |ax + by + c| / sqrt(a^2 + b^2)
        numerator = torch.abs(lines @ p1h.T)  # [L, S]
        denom = torch.sqrt(lines[:, 0:1] ** 2 + lines[:, 1:2] ** 2).clamp(min=1e-8)  # [L, 1]
        dist = numerator / denom  # [L, S]
        mask_b = (dist < epi_thresh).float()
        epi_masks.append(mask_b)
        # Debug: log mask stats to catch empty masks
        if mask_b.sum() == 0:
            from loguru import logger
            logger.warning(
                f"epi_mask is ALL ZEROS for batch {b}: "
                f"dist min={dist.min().item():.4f}, median={dist.median().item():.4f}, "
                f"max={dist.max().item():.4f}, thresh={epi_thresh}"
            )

    data['F_list'] = F_list
    data['epi_mask'] = torch.stack(epi_masks, dim=0)  # [B, L, S]

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
