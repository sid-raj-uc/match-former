import torch
import torch.nn as nn
import torch.nn.functional as F


class TripletCoarseLoss(nn.Module):
    """
    Triplet margin loss on the raw pre-softmax similarity matrix.

    For each GT positive pair (b, i, j_pos), picks the top 2*neg_per_pos most
    similar wrong columns in that row, randomly samples neg_per_pos of those as
    hard negatives, then computes:

        loss = mean( max(0, margin - sim[i, j_pos] + sim[i, j_neg]) )

    Once every triplet satisfies the margin, gradient is zero and features
    stabilize — avoiding the confidence collapse caused by focal loss on
    dual-softmax values that can never reach 1.0 over large grids.
    """
    def __init__(self, margin=1.0, neg_per_pos=10):
        super().__init__()
        self.margin = margin
        self.neg_per_pos = neg_per_pos

    def forward(self, sim_matrix, b_ids, i_ids, j_ids):
        """
        Args:
            sim_matrix [B, L, S]: raw pre-softmax similarity scores
            b_ids, i_ids, j_ids: GT match indices (batch, row in img0, col in img1)
        Returns:
            scalar loss
        """
        if len(b_ids) == 0:
            return sim_matrix.sum() * 0.0

        P = len(b_ids)
        N = self.neg_per_pos
        device = sim_matrix.device
        S = sim_matrix.shape[2]

        # Positive similarities: [P]
        sim_pos = sim_matrix[b_ids, i_ids, j_ids]  # [P]

        # For each positive, fetch its full row and zero out the GT column
        # so it cannot be picked as a hard negative
        rows = sim_matrix[b_ids, i_ids]  # [P, S]
        rows_masked = rows.clone()
        rows_masked[torch.arange(P, device=device), j_ids] = -1e9

        # Pick top 2*N hard negatives (by similarity) then randomly subsample N
        k = min(2 * N, S - 1)
        _, top_idx = torch.topk(rows_masked, k, dim=1)  # [P, k]
        rand_perm = torch.randperm(k, device=device)[:N]
        neg_idx = top_idx[:, rand_perm]  # [P, N]

        # Negative similarities: [P, N]
        sim_neg = rows[torch.arange(P, device=device).unsqueeze(1), neg_idx]

        # Triplet hinge loss: [P, N] → scalar
        loss = torch.clamp(self.margin - sim_pos.unsqueeze(1) + sim_neg, min=0.0)
        return loss.mean()


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
    def __init__(self, lambda_c=1.0, lambda_f=0.5, neg_per_pos=10):
        super().__init__()
        self.triplet = TripletCoarseLoss(margin=1.0, neg_per_pos=neg_per_pos)
        self.lambda_c = lambda_c
        self.lambda_f = lambda_f

    def forward(self, data):
        """
        Args:
            data (dict): output from Matchformer.forward(), augmented with supervision keys:
                'sim_matrix', 'spv_b_ids', 'spv_i_ids', 'spv_j_ids', 'expec_f', 'gt_mask'
        Returns:
            dict with 'loss', 'loss_c', 'loss_f'
        """
        sim_matrix = data['sim_matrix']
        spv_b = data['spv_b_ids']
        spv_i = data['spv_i_ids']
        spv_j = data['spv_j_ids']

        loss_c = self.triplet(sim_matrix, spv_b, spv_i, spv_j)
        loss_f = fine_loss(data)

        total = self.lambda_c * loss_c + self.lambda_f * loss_f

        return {
            'loss': total,
            'loss_c': loss_c.detach(),
            'loss_f': loss_f.detach() if isinstance(loss_f, torch.Tensor) else torch.tensor(0.0),
        }
