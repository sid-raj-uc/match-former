"""
Loss functions for fine-tuning MatchFormer with epipolar supervision.

L_coarse: Focal loss on the coarse confidence matrix.
    - Positive positions: GT match cells (spv_b_ids, spv_i_ids, spv_j_ids)
    - Everything else: negative
    
L_fine: L2 loss on fine-level sub-pixel offset prediction.
    - Only applied to GT-padded matches

Total: L = lambda_c * L_coarse + lambda_f * L_fine
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Binary focal loss: FL(p) = -alpha * (1-p)^gamma * log(p)
    Applied element-wise on the normalized confidence matrix.
    """
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, conf_matrix, b_ids, i_ids, j_ids, weight_pos=1.0):
        """
        Args:
            conf_matrix [B, L, S]: softmax-normalized confidence scores
            b_ids, i_ids, j_ids: GT match flat indices (batch, row in img0, col in img1)
            weight_pos: scalar multiplier for positive sample weight
        Returns:
            scalar loss
        """
        device = conf_matrix.device

        # Build binary GT label matrix (sparse, default 0)
        pos_mask = torch.zeros_like(conf_matrix)
        if len(b_ids) > 0:
            pos_mask[b_ids, i_ids, j_ids] = 1.0

        # Focal loss
        p = conf_matrix.clamp(1e-7, 1 - 1e-7)
        neg_mask = 1.0 - pos_mask

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
    
    Args:
        data (dict): must contain 'expec_f', 'spv_b_ids', 'spv_i_ids', 'spv_j_ids',
                     'mkpts0_c', 'mkpts1_c', and GT reprojection coords.
    Returns:
        scalar loss (0 if no GT fine matches available)
    """
    expec_f = data.get('expec_f')
    if expec_f is None or len(expec_f) == 0:
        return torch.tensor(0.0, requires_grad=True)

    # The fine loss is computed only on GT-padded matches (where gt_mask is True)
    gt_mask = data.get('gt_mask')
    if gt_mask is None or gt_mask.sum() == 0:
        return torch.tensor(0.0, device=expec_f.device)

    # expec_f[:, :2] are normalized coords; ground truth for GT-padded = (0, 0)
    # (GT padded matches have the coarse center as anchor, fine offset should be ~0)
    pred_coords = expec_f[gt_mask, :2]  # [N_gt, 2]
    gt_coords = torch.zeros_like(pred_coords)  # GT: center of window

    # Weight by inverse std for uncertainty-aware loss
    std = expec_f[gt_mask, 2].detach()
    weight = 1.0 / std.clamp(min=1e-4)

    loss = (weight * ((pred_coords - gt_coords) ** 2).sum(-1)).mean()
    return loss


class MatchFormerLoss(nn.Module):
    def __init__(self, lambda_c=1.0, lambda_f=0.5):
        super().__init__()
        self.focal = FocalLoss(alpha=0.25, gamma=2.0)
        self.lambda_c = lambda_c
        self.lambda_f = lambda_f

    def forward(self, data):
        """
        Args:
            data (dict): output from Matchformer.forward(), augmented with supervision keys:
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
