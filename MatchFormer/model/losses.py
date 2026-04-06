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

        pos_mask = torch.zeros_like(conf_matrix)
        if len(b_ids) > 0:
            pos_mask[b_ids, i_ids, j_ids] = 1.0

        if self.neg_per_pos > 0 and len(b_ids) > 0:
            neg_sample = self._sample_neg_mask(conf_matrix, b_ids, i_ids, j_ids)
            neg_mask = neg_sample.float() * (1.0 - pos_mask)
        else:
            neg_mask = 1.0 - pos_mask

        p = conf_matrix.clamp(1e-7, 1 - 1e-7)
        loss_pos = -self.alpha * (1 - p) ** self.gamma * torch.log(p) * pos_mask
        loss_neg = -(1 - self.alpha) * p ** self.gamma * torch.log(1 - p) * neg_mask

        loss = (weight_pos * loss_pos + loss_neg).sum() / (pos_mask.sum() + 1e-8)
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
