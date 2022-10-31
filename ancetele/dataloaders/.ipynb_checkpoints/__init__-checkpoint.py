import os
import sys 
sys.path.append("..")
from arguments import DataArguments
from transformers import (
    PreTrainedTokenizer, 
)
from .dense_dataset import (
    DenseTrainDataset,
    DenseEncodeDataset,
    DenseQPCollator,
    DenseEncodeCollator,
)
from .hf_dataset import (
    HFDataset, 
    HFQueryDataset, 
    HFCorpusDataset
)
from .dataset_utils import (
    TrainPreProcessor, 
    QueryPreProcessor, 
    CorpusPreProcessor
)

from .loader_utils import (EncodeCollator)




def get_train_dataset(
    tokenizer: PreTrainedTokenizer,
    data_args: DataArguments,
):
    
    ## Transformer load dataset
    train_dataset = HFDataset(
        tokenizer=tokenizer, 
        data_args=data_args,
        dataset_split="train",
        data_files=data_args.train_path,
        cache_dir=data_args.train_cache_dir,
    )

    return (
        DenseTrainDataset(
            data_args, 
            train_dataset.process(), 
            tokenizer), 
        None,
        DenseQPCollator
    )
    
    
def get_encode_dataset(
    tokenizer: PreTrainedTokenizer,
    data_args: DataArguments,
):
    
    ## Dense-Retriever
    if data_args.encode_is_qry:
        encode_dataset = HFQueryDataset(
            tokenizer=tokenizer, 
            data_args=data_args,
            dataset_split="encode",
            cache_dir=data_args.encode_in_path[0] + ".cache"
        )
    else:
        encode_dataset = HFCorpusDataset(
            tokenizer=tokenizer, 
            data_args=data_args,
            dataset_split="encode",
            cache_dir=data_args.encode_in_path[0] + ".cache"
        )
    return (
        DenseEncodeDataset(
            data_args,
            encode_dataset.process(data_args.encode_num_shard, data_args.encode_shard_index),
            tokenizer),
        DenseEncodeCollator
    )