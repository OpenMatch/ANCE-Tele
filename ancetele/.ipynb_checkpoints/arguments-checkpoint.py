import os
from dataclasses import dataclass, field
from typing import Optional, List
from transformers import TrainingArguments


@dataclass
class ModelArguments:
    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    target_model_path: str = field(
        default=None,
        metadata={"help": "Path to pretrained reranker target model"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None, metadata={"help": "Where do you want to store the pretrained models downloaded from s3"}
    )

    # modeling
    untie_encoder: bool = field(
        default=False,
        metadata={"help": "no weight sharing between qry passage encoders"}
    )

    # out projection
    add_pooler: bool = field(default=False)
    projection_in_dim: int = field(default=768)
    projection_out_dim: int = field(default=768)
        

@dataclass
class DataArguments:
    train_dir: str = field(
        default=None, metadata={"help": "Path to train directory"}
    )
    eval_dir: str = field(
        default=None, metadata={"help": "Path to eval directory"}
    )
    dataset_name: str = field(
        default=None, metadata={"help": "huggingface dataset name"}
    )
    passage_field_separator: str = field(default=' ')
    dataset_proc_num: int = field(
        default=12, metadata={"help": "number of proc used in dataset preprocess"}
    )
    train_n_passages: int = field(default=8)
    positive_passage_no_shuffle: bool = field(
        default=False, metadata={"help": "always use the first positive passage"})
    negative_passage_no_shuffle: bool = field(
        default=False, metadata={"help": "always use the first negative passages"})

    encode_in_path: List[str] = field(default=None, metadata={"help": "Path to data to encode"})
    encoded_save_path: str = field(default=None, metadata={"help": "where to save the encode"})
    encode_is_qry: bool = field(default=False)
    encode_num_shard: int = field(default=1)
    encode_shard_index: int = field(default=0)

    q_max_len: int = field(
        default=32,
        metadata={
            "help": "The maximum total input sequence length after tokenization for query. Sequences longer "
                    "than this will be truncated, sequences shorter will be padded."
        },
    )
    p_max_len: int = field(
        default=128,
        metadata={
            "help": "The maximum total input sequence length after tokenization for passage. Sequences longer "
                    "than this will be truncated, sequences shorter will be padded."
        },
    )
    train_cache_dir: Optional[str] = field(
        default=None, metadata={"help": "Where do you want to store the train data downloaded from huggingface, if None, repeated download"}
    )

    eval_cache_dir: Optional[str] = field(
        default=None, metadata={"help": "Where do you want to store the eval data downloaded from huggingface, if None, repeated download"}
    )
        
    ## split load
    split_load_data: bool = field(default=False)

    def __post_init__(self):
        if self.dataset_name is not None:
            info = self.dataset_name.split('/')
#             self.dataset_split = info[-1] if len(info) == 3 else 'train'
            self.dataset_name = "/".join(info[:-1]) if len(info) == 3 else '/'.join(info)
            self.dataset_language = 'default'
            if ':' in self.dataset_name:
                self.dataset_name, self.dataset_language = self.dataset_name.split(':')
        else:
            self.dataset_name = 'json'
#             self.dataset_split = 'train'
            self.dataset_language = 'default'
            
        if self.train_dir is not None:
            ## SS Modified --------------------
            self.train_cache_dir = os.path.join(self.train_dir, "cache") 
            ## SS Modified --------------------
            files = os.listdir(self.train_dir)
            self.train_path = [
                os.path.join(self.train_dir, f)
                for f in files
                if f.endswith('jsonl') or f.endswith('json')
            ]
        else:
            self.train_path = None
            
        if self.eval_dir is not None:
            ## SS Modified --------------------
            self.eval_cache_dir = os.path.join(self.eval_dir, "cache") 
            ## SS Modified --------------------
            files = os.listdir(self.eval_dir)
            self.eval_path = [
                os.path.join(self.eval_dir, f)
                for f in files
                if f.endswith('jsonl') or f.endswith('json')
            ]
        else:
            self.eval_path = None


@dataclass
class DenseTrainingArguments(TrainingArguments):
    warmup_ratio: float = field(default=0.1)
    negatives_x_device: bool = field(default=False, metadata={"help": "share negatives across devices"})
    do_encode: bool = field(default=False, metadata={"help": "run the encoding loop"})

    grad_cache: bool = field(default=False, metadata={"help": "Use gradient cache update"})
    gc_q_chunk_size: int = field(default=4)
    gc_p_chunk_size: int = field(default=32)
        
    ## SS added
    early_stop_step: int = field(default=-1)
    tensorboard: bool = field(default=False)
    