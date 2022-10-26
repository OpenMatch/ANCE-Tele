import json
import os
import copy
from dataclasses import dataclass
from typing import Optional, Dict

import torch
import torch.nn as nn
from torch import Tensor
import torch.distributed as dist

from transformers import AutoModel, PreTrainedModel
from transformers.modeling_outputs import ModelOutput

## ------- Modified by SS.
import sys
sys.path.append("..")
## ------- Modified by SS.

from tevatron.arguments import ModelArguments, DataArguments
from tevatron.arguments import DenseTrainingArguments as TrainingArguments


import logging
logger = logging.getLogger(__name__)


@dataclass
class CrossOutput(ModelOutput):
    qp_reps: Tensor = None
    loss: Tensor = None
    scores: Tensor = None


class ScorePooler(nn.Module):
    def __init__(
            self,
            input_dim: int = 768,
            output_dim: int = 1,
    ):
        super(ScorePooler, self).__init__()
        self.linear_qp = nn.Linear(input_dim, output_dim)
        self._config = {'input_dim': input_dim, 'output_dim': output_dim}

    def forward(self, qp: Tensor = None):
        if qp is not None:
            return self.linear_qp(qp)
        else:
            raise ValueError

    def load(self, ckpt_dir: str):
        if ckpt_dir is not None:
            _pooler_path = os.path.join(ckpt_dir, 'pooler.pt')
            if os.path.exists(_pooler_path):
                logger.info(f'Loading Pooler from {ckpt_dir}')
                state_dict = torch.load(os.path.join(ckpt_dir, 'pooler.pt'), map_location='cpu')
                self.load_state_dict(state_dict)
                return
        logger.info("Training Pooler from scratch")
        return

    def save_pooler(self, save_path):
        torch.save(self.state_dict(), os.path.join(save_path, 'pooler.pt'))
        with open(os.path.join(save_path, 'pooler_config.json'), 'w') as f:
            json.dump(self._config, f)


class CrossModel(nn.Module):
    def __init__(
            self,
            lm_qp: PreTrainedModel,
            pooler: nn.Module = None,
            model_args: ModelArguments = None,
            data_args: DataArguments = None,
            train_args: TrainingArguments = None,
    ):
        super().__init__()

        self.lm_qp = lm_qp
        self.pooler = pooler

        self.cross_entropy = nn.CrossEntropyLoss(reduction='mean')

        self.model_args = model_args
        self.train_args = train_args
        self.data_args = data_args


    def forward(
            self,
            query_passage: Dict[str, Tensor] = None,
    ):

        qp_hidden, qp_reps = self.encode_query_passage(query_passage)
#         qp_reps=qp_reps.view((
#             self.train_args.per_device_train_batch_size, 
#             self.data_args.train_n_passages,
#             -1,
#         ))
        
        scores = self.pooler(qp_reps).squeeze(-1) ## (bz * train_n_passages)
        
        if self.training:
            scores=scores.view((
                self.train_args.per_device_train_batch_size, 
                self.data_args.train_n_passages,
            ))
            target = torch.zeros(
                scores.size(0),
                device=scores.device,
                dtype=torch.long
            )
            loss = self.cross_entropy(scores, target)
            return loss
        else:
            return scores

#             return loss, (scores, qp_reps)

    def encode_query_passage(self, qry_passg):
        if qry_passg is None:
            return None, None
        qry_passg_out = self.lm_qp(**qry_passg, return_dict=True)
        qp_hidden = qry_passg_out.last_hidden_state
        qp_reps = qp_hidden[:, 0]
        return qp_hidden, qp_reps

    @staticmethod
    def build_pooler(model_args):
        pooler = ScorePooler(
            input_dim=model_args.projection_in_dim,
            output_dim=1,
        )
        pooler.load(model_args.model_name_or_path)
        return pooler

    @classmethod
    def build(
            cls,
            model_args: ModelArguments,
            data_args: DataArguments,
            train_args: TrainingArguments,
            **hf_kwargs,
    ):
        # load pre-trained
        lm_qp = AutoModel.from_pretrained(model_args.model_name_or_path, **hf_kwargs)            
        pooler = cls.build_pooler(model_args)

        model = cls(
            lm_qp=lm_qp,
            pooler=pooler,
            model_args=model_args,
            data_args=data_args,
            train_args=train_args
        )
        return model

    def save(self, output_dir: str):
        self.lm_qp.save_pretrained(output_dir)
        self.pooler.save_pooler(output_dir)


class CrossModelForInference(CrossModel):
    POOLER_CLS = ScorePooler

    def __init__(
            self,
            lm_qp: PreTrainedModel,
            pooler: nn.Module = None,
            **kwargs,
    ):
        nn.Module.__init__(self)
        self.lm_qp = lm_qp
        self.pooler = pooler

    @torch.no_grad()
    def encode_query_passage(self, qry_passg):
        return super(CrossModelForInference, self).encode_query_passage(qry_passg)

    def forward(
            self,
            query_passage: Dict[str, Tensor] = None,
    ):
        
        qp_hidden, qp_reps = self.encode_query_passage(query_passage)
        scores = self.pooler(qp_reps).squeeze(-1) ## (bz * train_n_passages)
        
        qp_hidden, qp_reps = self.encode_query_passage(query_passage)
        return scores

    @classmethod
    def build(
            cls,
            model_name_or_path: str = None,
            model_args: ModelArguments = None,
            data_args: DataArguments = None,
            train_args: TrainingArguments = None,
            **hf_kwargs,
    ):
        assert model_name_or_path is not None or model_args is not None
        if model_name_or_path is None:
            model_name_or_path = model_args.model_name_or_path

        logger.info(f'try loading tied weight')
        logger.info(f'loading model weight from {model_name_or_path}')
        lm_qp = AutoModel.from_pretrained(model_name_or_path, **hf_kwargs)

        pooler_weights = os.path.join(model_name_or_path, 'pooler.pt')
        pooler_config = os.path.join(model_name_or_path, 'pooler_config.json')
        
        logger.info(f'loading pooler weight and configuration')
        with open(pooler_config) as f:
            pooler_config_dict = json.load(f)
        pooler = cls.POOLER_CLS(**pooler_config_dict)
        pooler.load(model_name_or_path)

        model = cls(
            lm_qp=lm_qp,
            pooler=pooler
        )
        return model