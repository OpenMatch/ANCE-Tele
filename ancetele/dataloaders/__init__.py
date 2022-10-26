import os
from ..arguments import DataArguments
from transformers import (
    PreTrainedTokenizer, 
)
from .dense_dataset import (
    DenseTrainDataset,
    DenseEncodeDataset,
    DenseQPCollator,
    DenseEncodeCollator,
)
from .cross_dataset import (
    CrossTrainDataset, 
    CrossEvalDataset,
    CrossEncodeDataset,
    CrossQPCollator,
    CrossEncodeCollator,
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
    is_cross_encoder: bool
):
    
    ## Transformer load dataset
    train_dataset = HFDataset(
        tokenizer=tokenizer, 
        data_args=data_args,
        dataset_split="train",
        data_files=data_args.train_path,
        cache_dir=data_args.train_cache_dir,
    )

    if is_cross_encoder:
        
        eval_dataset = HFDataset(
            tokenizer=tokenizer, 
            data_args=data_args,
            dataset_split="eval",
            data_files=data_args.eval_path,
            cache_dir=data_args.eval_cache_dir,
        )
        
        return (
            CrossTrainDataset(
                data_args, 
                train_dataset.process(), 
                tokenizer),
            CrossEvalDataset(
                data_args, 
                eval_dataset.process(), 
                tokenizer),
            CrossQPCollator
        )

    else:
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
    is_cross_encoder: bool
):
    ## Cross-Encoder
    if is_cross_encoder:
        encode_in_dir = data_args.encode_in_path[0]
        files = os.listdir(encode_in_dir)
        
        data_args.encode_in_path = [
            os.path.join(encode_in_dir, f)
            for f in files
            if f.endswith('jsonl') or f.endswith('json')
        ]

        ## Transformer load dataset
        encode_dataset = HFDataset(
            tokenizer=tokenizer,
            data_args=data_args,
            dataset_split="train",
            data_files=data_args.encode_in_path,
            cache_dir=os.path.join(encode_in_dir, "cache"),
        )
        
        return (
            CrossEncodeDataset(
                data_args, 
                encode_dataset.process(data_args.encode_num_shard, data_args.encode_shard_index), 
                tokenizer),
            CrossEncodeCollator
        )
    
    ## Dense-Retriever
    else:
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