import os
from itertools import repeat
from typing import Dict, List, Tuple, Optional, Any, Union
from torch import nn
import torch.nn.functional as F
from transformers.trainer import Trainer

import torch
from torch.utils.data import DataLoader
import torch.distributed as dist

from ..losses import SimpleContrastiveLoss, DistributedContrastiveLoss

import logging
logger = logging.getLogger(__name__)


from torch.cuda.amp import autocast


class CrossTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super(CrossTrainer, self).__init__(*args, **kwargs)

    def _save(self, output_dir: Optional[str] = None):
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        logger.info("Saving model checkpoint to %s", output_dir)
        self.model.save(output_dir)
        
    def _prepare_inputs(
            self,
            inputs: Dict[str, Union[torch.Tensor, Any]]
    ):
        prepared = super()._prepare_inputs(inputs)
        return prepared

    def get_train_dataloader(self) -> DataLoader:
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")
        train_sampler = self._get_train_sampler()

        return DataLoader(
            self.train_dataset,
            batch_size=self.args.train_batch_size,
            sampler=train_sampler,
            collate_fn=self.data_collator,
            drop_last=True,
            num_workers=self.args.dataloader_num_workers,
        )

    def compute_loss(self, model, inputs):
        query_passage = inputs
        loss = model(query_passage=query_passage)
        return loss


    def prediction_step(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:

        inputs = self._prepare_inputs(inputs)
        query_passage = inputs

        labels = None
        loss = None
        with torch.no_grad():
            if self.use_amp:
                with autocast():
                    scores = model(query_passage=query_passage) ## eval_per_bz * train_n_psg
            else:
                scores = model(query_passage=query_passage)
                
            scores=scores.view((self.args.eval_batch_size, -1))
            
            logits = torch.argmax(scores, dim=1)
        
            labels = torch.zeros(
                logits.size(0),
                device=logits.device,
                dtype=torch.long
            )
            
            logits = (logits == labels).sum() / labels.size(0)
            
#             labels = torch.zeros(
#                 scores.size(0),
#                 device=scores.device,
#                 dtype=torch.long
#             )        
#             logits = F.cross_entropy(scores, labels, reduction='none')
        return (loss, logits, labels)
        
#     def prediction_step(
#         self,
#         model: nn.Module,
#         inputs: Dict[str, Union[torch.Tensor, Any]],
#         prediction_loss_only: bool,
#         ignore_keys: Optional[List[str]] = None,
#     ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        
#         loss, logits, labels = super(CrossTrainer, self).prediction_step(
#             model=model, 
#             inputs=inputs, 
#             prediction_loss_only=prediction_loss_only, 
#             ignore_keys=ignore_keys
#         )
        
# #         logits.view(self.args.eval_batch_size, -1)
# #         logits = torch.argmax(logits, dim=1)
        
#         labels = torch.ones(
#             logits.size(0),
#             device=logits.device,
#             dtype=torch.long
#         )
#         print("loss", loss)
#         print("logits", logits[:, 0].squeeze(-1))
#         print("labels", labels)
#         exit(0)
#         return loss, logits[:, 0].squeeze(-1), labels

    def training_step(self, *args):
        return super(CrossTrainer, self).training_step(*args)
    
