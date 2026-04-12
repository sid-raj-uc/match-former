import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Binary focal loss: FL(p) = -alpha * (1-p)^gamma * log(p)
    Applied element-wise on the normalized confidence matrix.

    neg_per_pos > 0: instead of computing loss on all ~23M negatives, sample
    neg_per_pos negatives per positive from the same row (same patch in image 0)
    and neg_per_pos from the same column (same patch in image 1). This keeps the
    negative-to-positive gradient ratio at 2*neg_per_pos:1 regardless of how many
    scenes are in the batch, preventing confidence collapse during multi-scene training.
    """
    def __init__(self, alpha=0.25, gamma=2.0, neg_per_pos=0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.neg_per_pos = neg_per_pos

    def _sample_neg_mask(self, conf_matrix, b_ids, i_ids, j_ids):
        B, L, S = conf_matrix.shape
        device = conf_matrix.device
        P, N = len(b_ids), self.neg_per_pos
        neg_sample = torch.zeros(B, L, S, dtype=torch.bool, device=device)

        # Row negatives: sample N cols from row i, excluding col j
        rand_cols = torch.randint(0, S - 1, (P, N), device=device)
        rand_cols[rand_cols >= j_ids.unsqueeze(1)] += 1
        neg_sample[
            b_ids.unsqueeze(1).expand(P, N).reshape(-1),
            i_ids.unsqueeze(1).expand(P, N).reshape(-1),
            rand_cols.reshape(-1),
        ] = True

        # Col negatives: sample N rows from col j, excluding row i
        rand_rows = torch.randint(0, L - 1, (P, N), device=device)
        rand_rows[rand_rows >= i_ids.unsqueeze(1)] += 1
        neg_sample[
            b_ids.unsqueeze(1).expand(P, N).reshape(-1),
            rand_rows.reshape(-1),
            j_ids.unsqueeze(1).expand(P, N).reshape(-1),
        ] = True

        return neg_sample

    def forward(self, conf_matrix, b_ids, i_ids, j_ids, weight_pos=1.0):
        device = conf_matrix.device

        # No GT matches → no meaningful loss
        if len(b_ids) == 0:
            return torch.tensor(0.0, device=device, requires_grad=True)

        pos_mask = torch.zeros_like(conf_matrix)
        pos_mask[b_ids, i_ids, j_ids] = 1.0

        if self.neg_per_pos > 0:
            neg_sample = self._sample_neg_mask(conf_matrix, b_ids, i_ids, j_ids)
            neg_mask = neg_sample.float() * (1.0 - pos_mask)
        else:
            neg_mask = 1.0 - pos_mask

        p = conf_matrix.clamp(1e-7, 1 - 1e-7)
        loss_pos = -self.alpha * (1 - p) ** self.gamma * torch.log(p) * pos_mask
        loss_neg = -(1 - self.alpha) * p ** self.gamma * torch.log(1 - p) * neg_mask

        loss = (weight_pos * loss_pos + loss_neg).sum() / pos_mask.sum()
        return loss


def fine_loss(data):
    """
    L2 loss on fine-level prediction offset.

    The model predicts 'expec_f': [M, 3] where [:, :2] is the normalized
    coordinate and [:, 2] is predicted std (used for confidence weighting).

    GT fine offset comes from the GT match projected into the fine window.
    """
    expec_f = data.get('expec_f')
    if expec_f is None or len(expec_f) == 0:
        return torch.tensor(0.0, requires_grad=True)

    gt_mask = data.get('gt_mask')
    if gt_mask is None or gt_mask.sum() == 0:
        return torch.tensor(0.0, device=expec_f.device)

    pred_coords = expec_f[gt_mask, :2]
    gt_coords = torch.zeros_like(pred_coords)

    std = expec_f[gt_mask, 2].detach()
    weight = 1.0 / std.clamp(min=1e-4)

    loss = (weight * ((pred_coords - gt_coords) ** 2).sum(-1)).mean()
    return loss


def epi_focal_loss(conf_matrix, epi_mask, alpha=0.25, gamma=2.0):
    """
    Search-reward focal loss using epipolar binary mask as pseudo-GT.
    No GT point correspondences needed — only camera geometry.

    Pushes confidence high near epipolar lines (mask=1),
    low far away (mask=0).

    Args:
        conf_matrix: [B, L, S] dual-softmax confidence matrix
        epi_mask:    [B, L, S] binary epipolar mask (1 = near epipolar line)
    """
    conf = conf_matrix.reshape(-1)
    mask = epi_mask.reshape(-1)

    p = conf.clamp(1e-7, 1 - 1e-7)
    # Focal loss: -alpha * (1-p)^gamma * log(p) * target
    #           - (1-alpha) * p^gamma * log(1-p) * (1-target)
    pos_loss = -alpha * (1 - p) ** gamma * torch.log(p) * mask
    neg_loss = -(1 - alpha) * p ** gamma * torch.log(1 - p) * (1 - mask)

    n_pos = mask.sum().clamp(min=1.0)
    loss = (pos_loss + neg_loss).sum() / n_pos
    return loss


def sampson_epipolar_loss(mkpts0, mkpts1, F_mat, b_ids):
    """
    SCENES-style Sampson epipolar distance loss.

    Args:
        mkpts0: [M, 2] predicted match coords in image 0 (pixels)
        mkpts1: [M, 2] predicted match coords in image 1 (pixels)
        F_mat:  list of 3x3 numpy arrays (one per batch element), or None entries
        b_ids:  [M] batch index for each match

    Returns:
        Scalar mean Sampson distance over all valid matches.
    """
    if mkpts0 is None or len(mkpts0) == 0:
        device = mkpts0.device if mkpts0 is not None else 'cpu'
        return torch.tensor(0.0, device=device, requires_grad=True)

    device = mkpts0.device
    total_loss = torch.tensor(0.0, device=device, requires_grad=True)
    count = 0

    for i, F_np in enumerate(F_mat):
        if F_np is None:
            continue
        mask = (b_ids == i)
        if mask.sum() == 0:
            continue

        p0 = mkpts0[mask]  # [n, 2]
        p1 = mkpts1[mask]  # [n, 2]

        # Homogeneous coords
        ones = torch.ones(p0.shape[0], 1, device=device)
        p0h = torch.cat([p0, ones], dim=1)  # [n, 3]
        p1h = torch.cat([p1, ones], dim=1)  # [n, 3]

        F_t = torch.tensor(F_np, dtype=torch.float32, device=device)

        # Sampson distance: (p1^T F p0)^2 / (||Fp0||_{1:2}^2 + ||F^T p1||_{1:2}^2)
        Fp0 = (F_t @ p0h.T).T     # [n, 3]
        Ftp1 = (F_t.T @ p1h.T).T  # [n, 3]
        num = (p1h * Fp0).sum(dim=1) ** 2  # (p1^T F p0)^2
        denom = Fp0[:, :2].pow(2).sum(dim=1) + Ftp1[:, :2].pow(2).sum(dim=1)
        sampson = num / (denom + 1e-8)

        total_loss = total_loss + sampson.sum()
        count += sampson.shape[0]

    if count == 0:
        return torch.tensor(0.0, device=device, requires_grad=True)

    return total_loss / count


def soft_sampson_loss(conf_matrix, F_list, grid_coords0, grid_coords1, margin=0.0):
    """
    Precision-steering Sampson loss — fully differentiable through conf_matrix.

    Step A: Soft match via confidence-weighted center of mass:
        p1_soft[b,i] = sum_j(conf[b,i,j] * grid_coords1[j]) / sum_j(conf[b,i,j])
    Step B: Sampson distance d_S^2 (squared pixel units), then sqrt → pixel units:
        d_S = sqrt(d_S^2 + eps)
    Step C: Apply geometric margin (dead zone, in pixels):
        d_S_margin = max(0, d_S - margin)
    Step D: Confidence-weighted normalization:
        loss = sum(C_max * d_S_margin) / (sum(C_max) + eps)

    The sqrt makes the penalty linear in pixel error (Huber/L1-like), preventing
    outlier matches from dominating the gradient. Combined with the margin, this
    yields a robust precision objective that doesn't drive confidence collapse.

    Args:
        conf_matrix:   [B, L, S] dual-softmax confidence matrix
        F_list:        list of [3, 3] numpy F matrices (one per batch element)
        grid_coords0:  [L, 2] pixel coords of each cell center in image 0
        grid_coords1:  [S, 2] pixel coords of each cell center in image 1
        margin:        geometric dead zone in PIXELS. Matches within `margin`
                       pixels of the epipolar line contribute zero loss.
    """
    B = conf_matrix.shape[0]
    device = conf_matrix.device

    total_loss = torch.tensor(0.0, device=device, requires_grad=True)
    count = 0

    for b in range(B):
        if b >= len(F_list) or F_list[b] is None:
            continue

        F_t = torch.tensor(F_list[b], dtype=torch.float32, device=device)
        conf = conf_matrix[b]  # [L, S]

        # Step A: soft match — center of mass in image 1
        p1_soft = torch.einsum('ls,sd->ld', conf, grid_coords1)  # [L, 2]
        conf_sum = conf.sum(dim=1, keepdim=True).clamp(min=1e-8)  # [L, 1]
        p1_soft = p1_soft / conf_sum  # [L, 2]

        # Homogeneous coords
        ones = torch.ones(grid_coords0.shape[0], 1, device=device)
        p0h = torch.cat([grid_coords0, ones], dim=1)  # [L, 3]
        p1h = torch.cat([p1_soft, ones], dim=1)        # [L, 3]

        # Step B: Sampson distance — squared pixel units, then sqrt → pixels
        Fp0 = (F_t @ p0h.T).T       # [L, 3]
        Ftp1 = (F_t.T @ p1h.T).T    # [L, 3]
        num = (p1h * Fp0).sum(dim=1) ** 2
        denom = Fp0[:, :2].pow(2).sum(dim=1) + Ftp1[:, :2].pow(2).sum(dim=1)
        sampson_sq = num / (denom + 1e-8)            # [L] squared pixels
        sampson = torch.sqrt(sampson_sq + 1e-8)      # [L] pixels (L1-like)

        # Step C: geometric margin (dead zone) — no penalty within `margin` pixels
        if margin > 0:
            sampson = torch.clamp(sampson - margin, min=0.0)

        # Step D: confidence-weighted normalization
        C_max = conf.max(dim=-1).values  # [L]
        total_loss = total_loss + (C_max * sampson).sum() / (C_max.sum() + 1e-8)
        count += 1

    if count == 0:
        return torch.tensor(0.0, device=device, requires_grad=True)

    return total_loss / count


class MatchFormerLoss(nn.Module):
    def __init__(self, lambda_c=1.0, lambda_f=0.0, neg_per_pos=0,
                 lambda_epi=0.7, sampson_margin=1.0):
        super().__init__()
        self.lambda_c = lambda_c
        self.lambda_f = lambda_f  # kept for logging; pose-only setup uses 0
        self.lambda_epi = lambda_epi
        self.sampson_margin = sampson_margin

    def _build_grid_coords(self, H, W, H_img, W_img, device):
        """Build pixel-space coordinates for coarse grid cell centers."""
        y, x = torch.meshgrid(
            torch.arange(H, device=device).float(),
            torch.arange(W, device=device).float(), indexing='ij')
        px = (x + 0.5) / W * W_img
        py = (y + 0.5) / H * H_img
        return torch.stack([px.flatten(), py.flatten()], dim=1)  # [H*W, 2]

    def forward(self, data):
        """
        Pose-only self-supervised loss:
            L_total  = lambda_c * L_coarse
            L_coarse = (1 - lambda_epi) * L_focal_epi + lambda_epi * L_sampson

        Both coarse terms depend ONLY on camera intrinsics + poses:
          - L_focal_epi: focal loss with binary epipolar mask as pseudo-GT
          - L_sampson:   confidence-weighted Sampson distance at soft matches

        Requires in data:
            'conf_matrix', 'epi_mask', 'F_list', 'hw0_c', 'hw1_c', 'hw0_i'
        """
        conf_matrix = data['conf_matrix']
        device = conf_matrix.device

        epi_mask = data['epi_mask']
        F_list = data['F_list']

        hw0_c = data['hw0_c']
        hw1_c = data['hw1_c']
        H_img, W_img = data['hw0_i']

        # Focal loss with epipolar mask as pseudo-GT (search reward)
        loss_focal = epi_focal_loss(conf_matrix, epi_mask)

        # Soft Sampson loss (precision steering)
        grid0 = self._build_grid_coords(hw0_c[0], hw0_c[1], H_img, W_img, device)
        grid1 = self._build_grid_coords(hw1_c[0], hw1_c[1], H_img, W_img, device)
        loss_sampson = soft_sampson_loss(
            conf_matrix, F_list, grid0, grid1,
            margin=self.sampson_margin,
        )

        # Combined: L_coarse = (1 - λ_epi) * L_focal_epi + λ_epi * L_sampson
        loss_c = (1 - self.lambda_epi) * loss_focal + self.lambda_epi * loss_sampson

        # Pose-only setup: no fine loss (would require depth-derived GT offsets)
        total = self.lambda_c * loss_c

        return {
            'loss': total,
            'loss_c': loss_c.detach(),
            'loss_f': torch.tensor(0.0, device=device),
            'loss_focal': loss_focal.detach(),
            'loss_sampson': loss_sampson.detach(),
        }
