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


def epipolar_coarse_loss(conf_matrix, epi_mask, alpha=0.25, gamma=2.0):
    """
    SCENES-style coarse loss: focal loss using epipolar mask as pseudo-GT.
    Pushes confidence high near epipolar lines, low far away.

    Args:
        conf_matrix: [B, L, S] dual-softmax confidence matrix
        epi_mask:    [B, L, S] Gaussian epipolar mask (values in [0, 1])
    """
    # Threshold mask to get binary target
    target = (epi_mask > 0.5).float()

    p = conf_matrix.clamp(1e-7, 1 - 1e-7)
    pos_loss = -alpha * (1 - p) ** gamma * torch.log(p) * target
    neg_loss = -(1 - alpha) * p ** gamma * torch.log(1 - p) * (1 - target)

    # Normalize by number of positives (like standard focal loss)
    n_pos = target.sum().clamp(min=1.0)
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


def soft_sampson_loss(conf_matrix, F_list, H0, W0, H1, W1, H_img=480, W_img=640):
    """
    Sampson epipolar loss on soft (confidence-weighted) match positions.
    Fully differentiable through conf_matrix — no argmax.

    For each query point i in image 0, computes:
        p1_soft[i] = sum_j(conf[i,j] * pos1[j]) / sum_j(conf[i,j])
        loss += sampson(p0[i], p1_soft[i], F)

    Args:
        conf_matrix: [B, L, S] dual-softmax confidence matrix
        F_list: list of 3x3 numpy F matrices (one per batch element)
        H0, W0: coarse grid dimensions for image 0
        H1, W1: coarse grid dimensions for image 1
    """
    B = conf_matrix.shape[0]
    device = conf_matrix.device

    # Build coarse grid positions in image pixel coords
    y0, x0 = torch.meshgrid(torch.arange(H0, device=device), torch.arange(W0, device=device), indexing='ij')
    pos0_x = (x0.float() + 0.5) / W0 * W_img  # [H0, W0]
    pos0_y = (y0.float() + 0.5) / H0 * H_img
    pos0 = torch.stack([pos0_x.flatten(), pos0_y.flatten()], dim=1)  # [L, 2]

    y1, x1 = torch.meshgrid(torch.arange(H1, device=device), torch.arange(W1, device=device), indexing='ij')
    pos1_x = (x1.float() + 0.5) / W1 * W_img
    pos1_y = (y1.float() + 0.5) / H1 * H_img
    pos1 = torch.stack([pos1_x.flatten(), pos1_y.flatten()], dim=1)  # [S, 2]

    total_loss = torch.tensor(0.0, device=device, requires_grad=True)
    count = 0

    for i in range(B):
        if i >= len(F_list) or F_list[i] is None:
            continue

        F_t = torch.tensor(F_list[i], dtype=torch.float32, device=device)
        conf = conf_matrix[i]  # [L, S]

        # Weighted average position in image 1 for each query point
        # conf_sum[j] avoids division by zero
        conf_sum = conf.sum(dim=1, keepdim=True).clamp(min=1e-8)  # [L, 1]
        weights = conf / conf_sum  # [L, S] — normalized weights per row

        # Soft match: p1_soft[i] = sum_j(w[i,j] * pos1[j])  — [L, 2]
        p1_soft = weights @ pos1  # [L, S] x [S, 2] = [L, 2]

        # Homogeneous coords
        ones = torch.ones(pos0.shape[0], 1, device=device)
        p0h = torch.cat([pos0, ones], dim=1)       # [L, 3]
        p1h = torch.cat([p1_soft, ones], dim=1)     # [L, 3]

        # Sampson distance
        Fp0 = (F_t @ p0h.T).T       # [L, 3]
        Ftp1 = (F_t.T @ p1h.T).T    # [L, 3]
        num = (p1h * Fp0).sum(dim=1) ** 2
        denom = Fp0[:, :2].pow(2).sum(dim=1) + Ftp1[:, :2].pow(2).sum(dim=1)
        sampson = num / (denom + 1e-8)

        total_loss = total_loss + sampson.mean()
        count += 1

    if count == 0:
        return torch.tensor(0.0, device=device, requires_grad=True)

    return total_loss / count


class MatchFormerLoss(nn.Module):
    def __init__(self, lambda_c=1.0, lambda_f=0.5, neg_per_pos=0):
        super().__init__()
        self.focal = FocalLoss(alpha=0.25, gamma=2.0, neg_per_pos=neg_per_pos)
        self.lambda_c = lambda_c
        self.lambda_f = lambda_f

    def forward(self, data):
        """
        Args:
            data (dict): model output augmented with supervision keys:
                'conf_matrix', 'spv_b_ids', 'spv_i_ids', 'spv_j_ids', 'expec_f', 'gt_mask'
        Returns:
            dict with 'loss', 'loss_c', 'loss_f'
        """
        conf_matrix = data['conf_matrix']
        spv_b = data['spv_b_ids']
        spv_i = data['spv_i_ids']
        spv_j = data['spv_j_ids']

        loss_c = self.focal(conf_matrix, spv_b, spv_i, spv_j)
        loss_f = fine_loss(data)

        total = self.lambda_c * loss_c + self.lambda_f * loss_f

        return {
            'loss': total,
            'loss_c': loss_c.detach(),
            'loss_f': loss_f.detach() if isinstance(loss_f, torch.Tensor) else torch.tensor(0.0),
        }
