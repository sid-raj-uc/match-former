import pprint
from loguru import logger
from pathlib import Path

import torch
import numpy as np
import pytorch_lightning as pl

from .matchformer import Matchformer
from .supervision import compute_supervision
from .losses import MatchFormerLoss
from .utils.metrics import (
    compute_symmetrical_epipolar_errors,
    compute_pose_errors,
    aggregate_metrics
)
from .utils.comm import gather
from .utils.misc import lower_config, flattenList
from .utils.profiler import PassThroughProfiler


class PL_LoFTR(pl.LightningModule):
    def __init__(self, config, pretrained_ckpt=None, profiler=None, dump_dir=None, freeze_backbone=True):
        super().__init__()
        # Misc
        self.config = config  # full config
        _config = lower_config(self.config)
        self.profiler = profiler or PassThroughProfiler()

        # Matcher: LoFTR
        self.matcher = Matchformer(config=_config['matchformer'])
        
        # Loss (only used in training)
        self.criterion = MatchFormerLoss(
            lambda_c=getattr(config, 'LOSS_LAMBDA_C', 1.0),
            lambda_f=getattr(config, 'LOSS_LAMBDA_F', 0.0),
            neg_per_pos=getattr(config, 'NEG_PER_POS', 0),
            lambda_epi=getattr(config, 'LAMBDA_EPI', 0.7),
            sampson_margin=getattr(config, 'SAMPSON_MARGIN', 1.0),
            focal_alpha=getattr(config, 'FOCAL_ALPHA', 0.5),
        )
        # Sampson annealing: hold lambda_epi=0 for `sampson_hold` steps, then
        # linearly ramp to the target over `sampson_ramp` steps.
        self.sampson_hold = getattr(config, 'SAMPSON_HOLD_STEPS', 1000)
        self.sampson_ramp = getattr(config, 'SAMPSON_RAMP_STEPS', 500)
        
        # Training hyperparameters
        self.lr = getattr(config, 'LR', 1e-4)
        self.weight_decay = getattr(config, 'WEIGHT_DECAY', 1e-4)
        self.warmup_steps = getattr(config, 'WARMUP_STEPS', 200)
        self.total_steps = getattr(config, 'TOTAL_STEPS', 10000)

        # Pretrained weights
        if pretrained_ckpt:
            ckpt = torch.load(pretrained_ckpt, map_location='cpu')
            state = ckpt.get('state_dict', ckpt)
            matcher_state = {k.replace('matcher.', ''): v for k, v in state.items() if k.startswith('matcher.')}
            if not matcher_state:
                matcher_state = {k: v for k, v in state.items() if not k.startswith(('epoch', 'global_step', 'pytorch-lightning', 'criterion'))}
            self.matcher.load_state_dict(matcher_state)
            logger.info(f"Load '{pretrained_ckpt}' as pretrained checkpoint")

        if freeze_backbone:
            # Freeze all parameters, then selectively unfreeze:
            #   - AttentionBlock3 + AttentionBlock4: the cross-attention stages that
            #     produce the coarse matching features
            #   - layer1_outconv + layer1_outconv2: the FPN head that outputs fine features
            for param in self.matcher.parameters():
                param.requires_grad = False

            trainable = [
                self.matcher.backbone.AttentionBlock3,
                self.matcher.backbone.AttentionBlock4,
                self.matcher.backbone.layer1_outconv,
                self.matcher.backbone.layer1_outconv2,
            ]
            for module in trainable:
                for param in module.parameters():
                    param.requires_grad = True

            # Freeze ALL BatchNorm running stats so pretrained statistics
            # are preserved.  BN stats are updated via EMA during train-mode
            # forward passes; on a small fine-tuning dataset they quickly
            # overwrite the rich pretrained distribution, corrupting features.
            for m in self.matcher.modules():
                if isinstance(m, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d, torch.nn.SyncBatchNorm)):
                    m.eval()           # keeps running_mean/var frozen during forward
                    m.weight.requires_grad = False
                    m.bias.requires_grad = False
            config._freeze_bn = True

            n_train = sum(p.numel() for p in self.matcher.parameters() if p.requires_grad)
            n_total = sum(p.numel() for p in self.matcher.parameters())
            logger.info(f"Trainable params: {n_train:,} / {n_total:,} "
                        f"({100*n_train/n_total:.1f}%) — "
                        f"AttentionBlock3, AttentionBlock4, fine FPN head (BN frozen)")
        else:
            n_train = sum(p.numel() for p in self.matcher.parameters())
            n_total = n_train
            logger.info(f"Trainable params: {n_train:,} / {n_total:,} (100.0%) — all weights unfrozen")
        
        # Testing
        self.dump_dir = dump_dir

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        eta_min = getattr(self.config, 'ETA_MIN', 1e-6)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.total_steps, eta_min=eta_min
        )
        return {
            'optimizer': optimizer,
            'lr_scheduler': {'scheduler': scheduler, 'interval': 'step'}
        }

    def _update_sampson_anneal(self):
        """
        Lambda_epi annealing schedule:
          [0, hold)                   → 0
          [hold, hold + ramp)         → linear ramp 0 → target
          [hold + ramp, ∞)            → target
        """
        step = self.global_step
        target = self.criterion.lambda_epi_target
        if step < self.sampson_hold:
            self.criterion.lambda_epi = 0.0
        elif self.sampson_ramp > 0 and step < self.sampson_hold + self.sampson_ramp:
            frac = (step - self.sampson_hold) / float(self.sampson_ramp)
            self.criterion.lambda_epi = target * frac
        else:
            self.criterion.lambda_epi = target

    def training_step(self, batch, batch_idx):
        # Sampson annealing: update lambda_epi based on global step
        self._update_sampson_anneal()

        # Step 1: temporarily run in eval mode to populate hw0_c, hw1_c
        # without triggering coarse_matching's training-mode GT padding
        # (which needs spv_b_ids that don't exist yet)
        self.matcher.eval()
        with torch.no_grad():
            self.matcher(batch)
        self.matcher.train()
        # Keep BN in eval mode so running stats stay frozen
        if getattr(self.config, '_freeze_bn', False):
            for m in self.matcher.modules():
                if isinstance(m, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d, torch.nn.SyncBatchNorm)):
                    m.eval()

        # Step 2: Compute supervision labels (populates spv_b_ids/i_ids/j_ids)
        compute_supervision(batch, self.config)

        # Step 3: Clear keys that the forward will re-populate
        for key in ['conf_matrix', 'sim_matrix', 'b_ids', 'i_ids', 'j_ids', 'gt_mask',
                    'm_bids', 'mkpts0_c', 'mkpts1_c', 'mconf',
                    'expec_f', 'mkpts0_f', 'mkpts1_f']:
            batch.pop(key, None)

        # Step 4: Real training forward (coarse_matching now finds spv_* keys)
        self.matcher(batch)

        # Step 5: Compute losses
        losses = self.criterion(batch)

        self.log('train/loss',   losses['loss'],   on_step=True, on_epoch=True, prog_bar=True)
        self.log('train/loss_c', losses['loss_c'], on_step=True, on_epoch=True)
        self.log('train/loss_f', losses['loss_f'], on_step=True, on_epoch=True)
        self.log('train/loss_focal',   losses['loss_focal'],   on_step=True, on_epoch=True)
        self.log('train/loss_sampson', losses['loss_sampson'], on_step=True, on_epoch=True)
        self.log('train/lambda_epi',   float(self.criterion.lambda_epi),
                 on_step=True, on_epoch=False)

        # GT supervision — how many ground truth matches were found this batch
        spv_b = batch.get('spv_b_ids')
        if spv_b is not None:
            self.log('train/num_gt_matches', float(len(spv_b)), on_step=True, on_epoch=False)

        # Predicted matches that survived mutual nearest-neighbour + threshold
        b_ids = batch.get('b_ids')
        if b_ids is not None:
            self.log('train/num_matches', float(len(b_ids)), on_step=True, on_epoch=False)

        # Confidence matrix health — collapse detector
        conf = batch.get('conf_matrix')
        if conf is not None:
            self.log('train/conf_max',  conf.max().item(),  on_step=True, on_epoch=False)
            self.log('train/conf_mean', conf.mean().item(), on_step=True, on_epoch=False)

        # Epipolar mask health
        epi_mask = batch.get('epi_mask')
        if epi_mask is not None:
            self.log('train/epi_mask_frac', epi_mask.mean().item(), on_step=True, on_epoch=False)

        return losses['loss']

    def validation_step(self, batch, batch_idx):
        # We need to compute supervision first so that self.criterion works
        from .supervision import compute_supervision
        compute_supervision(batch, self.config)
        self.matcher(batch)
        
        losses = self.criterion(batch)
        
        self.log('val/loss',   losses['loss'],   on_step=False, on_epoch=True, prog_bar=True)
        self.log('val/loss_c', losses['loss_c'], on_step=False, on_epoch=True)
        self.log('val/loss_f', losses['loss_f'], on_step=False, on_epoch=True)
        self.log('val/loss_focal',   losses['loss_focal'],   on_step=False, on_epoch=True)
        self.log('val/loss_sampson', losses['loss_sampson'], on_step=False, on_epoch=True)

        spv_b = batch.get('spv_b_ids')
        if spv_b is not None:
            self.log('val/num_gt_matches', float(len(spv_b)), on_epoch=True)

        mkpts0 = batch.get('mkpts0_f')
        if mkpts0 is not None and len(mkpts0) > 0:
            self.log('val/num_matches', float(len(mkpts0)), on_epoch=True)
        
        return losses['loss']
        
    
    def _compute_metrics(self, batch):
        with self.profiler.profile("Copmute metrics"):
            compute_symmetrical_epipolar_errors(batch)  # compute epi_errs for each match
            compute_pose_errors(batch, self.config)  # compute R_errs, t_errs, pose_errs for each pair

            rel_pair_names = list(zip(*batch['pair_names']))
            bs = batch['image0'].size(0)
            metrics = {
                # to filter duplicate pairs caused by DistributedSampler
                'identifiers': ['#'.join(rel_pair_names[b]) for b in range(bs)],
                'epi_errs': [batch['epi_errs'][batch['m_bids'] == b].cpu().numpy() for b in range(bs)],
                'R_errs': batch['R_errs'],
                't_errs': batch['t_errs'],
                'inliers': batch['inliers']}
            ret_dict = {'metrics': metrics}
        return ret_dict, rel_pair_names
    

    def test_step(self, batch, batch_idx):
        with self.profiler.profile("LoFTR"):
            self.matcher(batch)

        ret_dict, rel_pair_names = self._compute_metrics(batch)

        with self.profiler.profile("dump_results"):
            if self.dump_dir is not None:
                # dump results for further analysis
                keys_to_save = {'mkpts0_f', 'mkpts1_f', 'mconf', 'epi_errs'}
                pair_names = list(zip(*batch['pair_names']))
                bs = batch['image0'].shape[0]
                dumps = []
                for b_id in range(bs):
                    item = {}
                    mask = batch['m_bids'] == b_id
                    item['pair_names'] = pair_names[b_id]
                    item['identifier'] = '#'.join(rel_pair_names[b_id])
                    for key in keys_to_save:
                        item[key] = batch[key][mask].cpu().numpy()
                    for key in ['R_errs', 't_errs', 'inliers']:
                        item[key] = batch[key][b_id]
                    dumps.append(item)
                ret_dict['dumps'] = dumps

        return ret_dict

    def test_epoch_end(self, outputs):
        # metrics: dict of list, numpy
        _metrics = [o['metrics'] for o in outputs]
        metrics = {k: flattenList(gather(flattenList([_me[k] for _me in _metrics]))) for k in _metrics[0]}

        # [{key: [{...}, *#bs]}, *#batch]
        if self.dump_dir is not None:
            Path(self.dump_dir).mkdir(parents=True, exist_ok=True)
            _dumps = flattenList([o['dumps'] for o in outputs])  # [{...}, #bs*#batch]
            dumps = flattenList(gather(_dumps))  # [{...}, #proc*#bs*#batch]
            logger.info(f'Prediction and evaluation results will be saved to: {self.dump_dir}')

        if self.trainer.global_rank == 0:
            print(self.profiler.summary())
            val_metrics_4tb = aggregate_metrics(metrics, self.config.TRAINER.EPI_ERR_THR)
            logger.info('\n' + pprint.pformat(val_metrics_4tb))
            if self.dump_dir is not None:
                np.save(Path(self.dump_dir) / 'LoFTR_pred_eval', dumps)