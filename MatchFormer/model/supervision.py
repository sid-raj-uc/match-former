import torch
import numpy as np


@torch.no_grad()
def compute_supervision(data, config):
    """
    Self-supervised epipolar supervision — uses ONLY camera intrinsics + poses.
    No depth, no GT correspondences.

    Expects in data:
        'K'      [B, 3, 3]  camera intrinsics
        'T0'     [B, 4, 4]  camera-to-world pose
        'T1'     [B, 4, 4]  camera-to-world pose
        'hw0_c'  (H_c, W_c) coarse feature grid size
        'hw0_i'  (H_i, W_i) input image size

    Adds to data:
        'F_list'     list of B [3,3] fundamental matrices
        'epi_mask'   [B, L, S] binary mask: cell (i,j)=1 if image-1 grid
                     point j is within EPI_THRESH pixels of image-0 grid
                     point i's epipolar line
        'spv_b_ids'  empty tensor (kept for coarse_matching compatibility)
        'spv_i_ids'  empty tensor
        'spv_j_ids'  empty tensor
    """
    device = data['image0'].device
    B = data['image0'].shape[0]

    H_img, W_img = data['hw0_i']
    H_c, W_c = data['hw0_c']

    K = data['K'].to(device)    # [B, 3, 3]
    T0 = data['T0'].to(device)  # [B, 4, 4]
    T1 = data['T1'].to(device)  # [B, 4, 4]

    epi_thresh = getattr(config, 'EPI_THRESH', 2.0)  # pixels

    # Grid positions in pixel coords for Image 0 and Image 1 (coarse grid centers)
    y0g, x0g = torch.meshgrid(
        torch.arange(H_c, device=device).float(),
        torch.arange(W_c, device=device).float(), indexing='ij')
    pos0_x = (x0g + 0.5) / W_c * W_img
    pos0_y = (y0g + 0.5) / H_c * H_img
    p0h = torch.stack([pos0_x.flatten(), pos0_y.flatten(),
                       torch.ones(H_c * W_c, device=device)], dim=1)  # [L, 3]

    y1g, x1g = torch.meshgrid(
        torch.arange(H_c, device=device).float(),
        torch.arange(W_c, device=device).float(), indexing='ij')
    pos1_x = (x1g + 0.5) / W_c * W_img
    pos1_y = (y1g + 0.5) / H_c * H_img
    p1h = torch.stack([pos1_x.flatten(), pos1_y.flatten(),
                       torch.ones(H_c * W_c, device=device)], dim=1)  # [S, 3]

    F_list = []
    epi_masks = []

    for b in range(B):
        T0_np = T0[b].cpu().float().numpy()
        T1_np = T1[b].cpu().float().numpy()
        K_np = K[b].cpu().float().numpy()

        if not (np.isfinite(T0_np).all() and np.isfinite(T1_np).all()):
            F_list.append(None)
            epi_masks.append(torch.zeros(H_c * W_c, H_c * W_c, device=device))
            continue

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

        # Epipolar mask: point-to-line distance |aᵀp| / sqrt(a² + b²)
        F_t = torch.tensor(F_mat, dtype=torch.float32, device=device)
        lines = p0h @ F_t.T  # [L, 3] — epipolar lines in image 1
        numerator = torch.abs(lines @ p1h.T)  # [L, S]
        denom = torch.sqrt(lines[:, 0:1] ** 2 + lines[:, 1:2] ** 2).clamp(min=1e-8)
        dist = numerator / denom  # [L, S]
        mask_b = (dist < epi_thresh).float()
        epi_masks.append(mask_b)

        if mask_b.sum() == 0:
            from loguru import logger
            logger.warning(
                f"epi_mask is ALL ZEROS for batch {b}: "
                f"dist min={dist.min().item():.4f}, median={dist.median().item():.4f}, "
                f"max={dist.max().item():.4f}, thresh={epi_thresh}"
            )

    data['F_list'] = F_list
    data['epi_mask'] = torch.stack(epi_masks, dim=0)  # [B, L, S]

    # Empty spv_* — no GT correspondences in the pose-only setting.
    # coarse_matching guards on `len(spv_b_ids) > 0` so this is safe.
    empty = torch.zeros(0, dtype=torch.long, device=device)
    data.update({
        'spv_b_ids': empty,
        'spv_i_ids': empty,
        'spv_j_ids': empty,
    })
