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
    def __init__(self, config, pretrained_ckpt=None, profiler=None, dump_dir=None):
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
            lambda_f=getattr(config, 'LOSS_LAMBDA_F', 0.5),
        )
        
        # Training hyperparameters
        self.lr = getattr(config, 'LR', 1e-4)
        self.weight_decay = getattr(config, 'WEIGHT_DECAY', 1e-4)
        self.warmup_steps = getattr(config, 'WARMUP_STEPS', 200)
        self.total_steps = getattr(config, 'TOTAL_STEPS', 10000)

        # Pretrained weights
        if pretrained_ckpt:
            self.matcher.load_state_dict({k.replace('matcher.',''):v  for k,v in torch.load(pretrained_ckpt, map_location='cpu').items()})
            logger.info(f"Load '{pretrained_ckpt}' as pretrained checkpoint")
        
        # Testing
        self.dump_dir = dump_dir

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.total_steps, eta_min=1e-6
        )
        return {
            'optimizer': optimizer,
            'lr_scheduler': {'scheduler': scheduler, 'interval': 'step'}
        }

    def training_step(self, batch, batch_idx):
        # Step 1: temporarily run in eval mode to populate hw0_c, hw1_c
        # without triggering coarse_matching's training-mode GT padding
        # (which needs spv_b_ids that don't exist yet)
        self.matcher.eval()
        with torch.no_grad():
            self.matcher(batch)
        self.matcher.train()

        # Step 2: Compute supervision labels (populates spv_b_ids/i_ids/j_ids)
        compute_supervision(batch, self.config)

        # Step 3: Clear keys that the forward will re-populate
        for key in ['conf_matrix', 'b_ids', 'i_ids', 'j_ids', 'gt_mask',
                    'm_bids', 'mkpts0_c', 'mkpts1_c', 'mconf',
                    'expec_f', 'mkpts0_f', 'mkpts1_f']:
            batch.pop(key, None)

        # Step 4: Real training forward (coarse_matching now finds spv_* keys)
        self.matcher(batch)

        # Step 5: Compute losses
        losses = self.criterion(batch)

        self.log('train/loss',   losses['loss'],   on_step=True, prog_bar=True)
        self.log('train/loss_c', losses['loss_c'], on_step=True)
        self.log('train/loss_f', losses['loss_f'], on_step=True)
        return losses['loss']

    def validation_step(self, batch, batch_idx):
        self.matcher(batch)
        # Compute mean GT reprojection error over predicted matches
        from gt_epipolar import compute_fundamental_matrix
        import numpy as np
        mkpts0 = batch.get('mkpts0_f')
        mkpts1 = batch.get('mkpts1_f')
        if mkpts0 is not None and len(mkpts0) > 0:
            self.log('val/num_matches', float(len(mkpts0)), on_epoch=True)
        return {}
        
    
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