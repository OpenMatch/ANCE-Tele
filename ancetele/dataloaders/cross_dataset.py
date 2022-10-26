import random
from dataclasses import dataclass
from typing import List, Tuple

import datasets
from tqdm import tqdm
from torch.utils.data import Dataset
from transformers import (
    PreTrainedTokenizer, 
    BatchEncoding, 
    DataCollatorWithPadding
)

from ..arguments import DataArguments
from ..trainers import CrossTrainer

import logging
logger = logging.getLogger(__name__)


class CrossTrainDataset(Dataset):
    def __init__(
            self,
            data_args: DataArguments,
            dataset: datasets.Dataset,
            tokenizer: PreTrainedTokenizer,
            trainer: CrossTrainer = None,
    ):
        self.examples = dataset
        self.tok = tokenizer
        self.trainer = trainer

        self.data_args = data_args
        self.total_len = len(self.examples)

    def create_one_example(self, text_encoding: List[int]):
        item = self.tok.encode_plus(
            text_encoding,
            truncation='only_first',
            max_length=(self.data_args.q_max_len + 1 + self.data_args.p_max_len),
            padding=False,
            return_attention_mask=False,
            return_token_type_ids=False,
        )
        return item

    def __len__(self):
        return self.total_len

    def __getitem__(self, item) -> Tuple[BatchEncoding, List[BatchEncoding]]:
        group = self.examples[item]
        
        qry = group['query']
        group_positives = group['positives']
        group_negatives = group['negatives']
        
        epoch = int(self.trainer.state.epoch)
        _hashed_seed = hash(item + self.trainer.args.seed)

        ## Positive Passage (Single)
        ## Diff positive passage per epoch
        if self.data_args.positive_passage_no_shuffle:
            pos_psg = group_positives[0]
        else:
            pos_psg = group_positives[(_hashed_seed + epoch) % len(group_positives)]
            
        ## Negative Passage (List)
        negative_size = self.data_args.train_n_passages - 1
        if len(group_negatives) < negative_size:
            negs = random.choices(group_negatives, k=negative_size)
        elif self.data_args.train_n_passages == 1:
            negs = []
        elif self.data_args.negative_passage_no_shuffle:
            negs = group_negatives[:negative_size]
        else:
            _offset = epoch * negative_size % len(group_negatives)
            negs = [x for x in group_negatives]
            random.Random(_hashed_seed).shuffle(negs)
            negs = negs * 2
            negs = negs[_offset: _offset + negative_size]
            
        ## Concate qry + psg
        encoded_query_passages = []
        encoded_query_passages.append(self.create_one_example(
            qry + [self.tok.sep_token_id] + pos_psg
        ))
        
        for neg_psg in negs:
            encoded_query_passages.append(self.create_one_example(
                qry + [self.tok.sep_token_id] + neg_psg
            ))        
        return encoded_query_passages

    

class CrossEvalDataset(Dataset):
    def __init__(
            self,
            data_args: DataArguments,
            dataset: datasets.Dataset,
            tokenizer: PreTrainedTokenizer,
    ):
        self.examples = dataset
        self.tok = tokenizer

        self.data_args = data_args
        self.total_len = len(self.examples)

    def create_one_example(self, text_encoding: List[int]):
        item = self.tok.encode_plus(
            text_encoding,
            truncation='only_first',
            max_length=(self.data_args.q_max_len + 1 + self.data_args.p_max_len),
            padding=False,
            return_attention_mask=False,
            return_token_type_ids=False,
        )
        return item

    def __len__(self):
        return self.total_len

    def __getitem__(self, item) -> Tuple[BatchEncoding, List[BatchEncoding]]:
        group = self.examples[item]
        
        qry = group['query']
        group_positives = group['positives']
        group_negatives = group['negatives']
        
        ## Positive Passage (Single)
        pos_psg = group_positives[0]
        
        ## Negative Passage (List)
        negative_size = 30
        negs = group_negatives[:negative_size]
            
        ## Concate qry + psg
        encoded_query_passages = []
        encoded_query_passages.append(self.create_one_example(
            qry + [self.tok.sep_token_id] + pos_psg
        ))
        
        for neg_psg in negs:
            encoded_query_passages.append(self.create_one_example(
                qry + [self.tok.sep_token_id] + neg_psg
            ))        
        return encoded_query_passages

    


@dataclass
class CrossQPCollator(DataCollatorWithPadding):
    """
    Wrapper that does conversion from List[Tuple[encode_qry, encode_psg]] to List[qry], List[psg]
    and pass batch separately to the actual collator.
    Abstract out data detail for the model.
    """ 
    max_q_len: int = 32
    max_p_len: int = 128

    def __call__(self, features):
        
        qd = features
        
        if isinstance(qd[0], list):
            qd = sum(qd, [])

        qd_collated = self.tokenizer.pad(
            qd,
            padding='max_length',
            max_length=(self.max_q_len + 1 + self.max_p_len),
            return_tensors="pt",
        )
        """
        Dict = {"input_ids", "attention_mask"}
        
        """
        return qd_collated

        
        
class CrossEncodeDataset(Dataset):
    def __init__(
            self,
            data_args: DataArguments,
            dataset: datasets.Dataset,
            tokenizer: PreTrainedTokenizer,
    ):
        self.examples = dataset
        self.tok = tokenizer

        self.data_args = data_args
        self.total_len = len(self.examples)

    def create_one_example(self, text_encoding: List[int]):
        item = self.tok.encode_plus(
            text_encoding,
            truncation='only_first',
            max_length=(self.data_args.q_max_len + 1 + self.data_args.p_max_len),
            padding=False,
            return_attention_mask=False,
            return_token_type_ids=False,
        )
        return item

    def __len__(self):
        return self.total_len

    def __getitem__(self, item) -> Tuple[BatchEncoding, List[BatchEncoding]]:
        group = self.examples[item]
        
        qry = group['query']
        group_positives = group['positives']
        group_negatives = group['negatives']
        
        encoded_query_passages = []
        for passage in group_positives + group_negatives:
            encoded_query_passages.append(self.create_one_example(
                qry + [self.tok.sep_token_id] + passage
            ))  
        return encoded_query_passages

    
@dataclass
class CrossEncodeCollator(DataCollatorWithPadding):
    def __call__(self, features):
        qd = features

        if isinstance(qd[0], list):
            qd = sum(qd, [])
            
        collated_features = super().__call__(qd)
        return collated_features

    
# class CrossEncodeDataset(Dataset):
#     def __init__(
#             self,
#             data_args: DataArguments,
#             dataset: datasets.Dataset,
#             tokenizer: PreTrainedTokenizer,
#     ):
#         self.create_dataset(dataset)
#         self.tok = tokenizer

#         self.data_args = data_args
#         self.total_len = len(self.examples)
        
#     def create_dataset(self, dataset):
#         examples = []
#         qid2query = []
#         qid2passage = {}
#         for qid, data in enumerate(tqdm(dataset)):
#             qry = data['query']
#             group_positives = data['positives']
#             group_negatives = data['negatives']
            
#             qid2query.append(qry)
#             qid2passage[qid] = {'pos': []}
#             for pid, passage in enumerate(group_positives):
#                 examples.append({'query': qry, 'passage': passage, 'id': '-'.join(['pos', str(qid), str(pid)])})
#                 qid2passage[qid]['pos'].append(passage)
                
#             qid2passage[qid] = {'neg': []}
#             for pid, passage in enumerate(group_negatives):
#                 examples.append({'query': qry, 'passage': passage, 'id': '-'.join(['neg', str(qid), str(pid)])})
#                 qid2passage[qid]['neg'].append(passage)            
            
#         self.examples = examples
#         self.qid2query = qid2query
#         self.qid2passage = qid2passage


#     def create_one_example(self, text_encoding: List[int]):
#         item = self.tok.encode_plus(
#             text_encoding,
#             truncation='only_first',
#             max_length=(self.data_args.q_max_len + 1 + self.data_args.p_max_len),
#             padding=False,
#             return_attention_mask=False,
#             return_token_type_ids=False,
#         )
#         return item

#     def __len__(self):
#         return self.total_len

#     def __getitem__(self, item) -> Tuple[BatchEncoding, List[BatchEncoding]]:
#         group = self.examples[item]
        
#         qry = group['query']
#         passage = group['passage']
#         id_ = group['id']
            
#         ## Concate qry + psg
#         encoded_query_passage = self.create_one_example(
#             qry + [self.tok.sep_token_id] + passage
#         )
         
#         return id_, encoded_query_passage

    
