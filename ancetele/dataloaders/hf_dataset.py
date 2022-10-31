import os
from datasets import load_dataset, concatenate_datasets
from transformers import PreTrainedTokenizer
from .dataset_utils import TrainPreProcessor, QueryPreProcessor, CorpusPreProcessor

import sys 
sys.path.append("..")
from arguments import DataArguments

DEFAULT_PROCESSORS = [TrainPreProcessor, QueryPreProcessor, CorpusPreProcessor]
PROCESSOR_INFO = {
    'json': [None, None, None]
}


class HFDataset:
    def __init__(
        self, 
        tokenizer: PreTrainedTokenizer, 
        data_args: DataArguments, 
        dataset_split: str,
        data_files: str,
        cache_dir: str
    ):
        
        ## ********************************************************
        if data_args.split_load_data:
            dataset_list = []
            for filepath in data_files:
                dataset_list.append(
                    load_dataset(
                        data_args.dataset_name,
                        data_args.dataset_language,
                        data_files=[filepath], 
                        cache_dir=os.path.join(filepath+".cache"),
                    )[dataset_split]
                )
                
            self.dataset = concatenate_datasets(dataset_list)
        ## ********************************************************
        else:
            
            if data_files:
                data_files = {dataset_split: data_files}
                ## {"train/eval": [filepath_1, filepath_2, ...]}

            self.dataset = load_dataset(
                data_args.dataset_name,
                data_args.dataset_language,
                data_files=data_files, 
                cache_dir=cache_dir
            )[dataset_split]

        self.preprocessor = PROCESSOR_INFO[data_args.dataset_name][0] if data_args.dataset_name in PROCESSOR_INFO \
            else DEFAULT_PROCESSORS[0] ## None
        
        self.tokenizer = tokenizer
        self.q_max_len = data_args.q_max_len
        self.p_max_len = data_args.p_max_len
        self.proc_num = data_args.dataset_proc_num
        self.neg_num = data_args.train_n_passages - 1
        self.separator = getattr(self.tokenizer, data_args.passage_field_separator, data_args.passage_field_separator)

    def process(self, shard_num=1, shard_idx=0):
        self.dataset = self.dataset.shard(shard_num, shard_idx)
        if self.preprocessor is not None:
            self.dataset = self.dataset.map(
                self.preprocessor(self.tokenizer, self.q_max_len, self.p_max_len, self.separator),
                batched=False,
                num_proc=self.proc_num,
                remove_columns=self.dataset.column_names,
                desc="Running tokenizer on train dataset",
            )
        return self.dataset


class HFQueryDataset:
    def __init__(
        self, 
        tokenizer: PreTrainedTokenizer, 
        data_args: DataArguments, 
        cache_dir: str,
        dataset_split: str,
    ):
        data_files = data_args.encode_in_path
        if data_files:
            data_files = {dataset_split: data_files}
        self.dataset = load_dataset(data_args.dataset_name,
                                    data_args.dataset_language,
                                    data_files=data_files, cache_dir=cache_dir)[dataset_split]
        self.preprocessor = PROCESSOR_INFO[data_args.dataset_name][1] if data_args.dataset_name in PROCESSOR_INFO \
            else DEFAULT_PROCESSORS[1]
        self.tokenizer = tokenizer
        self.q_max_len = data_args.q_max_len
        self.proc_num = data_args.dataset_proc_num

    def process(self, shard_num=1, shard_idx=0):
        self.dataset = self.dataset.shard(shard_num, shard_idx)
        if self.preprocessor is not None:
            self.dataset = self.dataset.map(
                self.preprocessor(self.tokenizer, self.q_max_len),
                batched=False,
                num_proc=self.proc_num,
                remove_columns=self.dataset.column_names,
                desc="Running tokenization",
            )
        return self.dataset


class HFCorpusDataset:
    def __init__(
        self, 
        tokenizer: PreTrainedTokenizer, 
        data_args: DataArguments, 
        cache_dir: str,
        dataset_split: str,
    ):
        data_files = data_args.encode_in_path
        if data_files:
            data_files = {dataset_split: data_files}
        self.dataset = load_dataset(data_args.dataset_name,
                                    data_args.dataset_language,
                                    data_files=data_files, cache_dir=cache_dir)[dataset_split]
        script_prefix = data_args.dataset_name
        if script_prefix.endswith('-corpus'):
            script_prefix = script_prefix[:-7]
        self.preprocessor = PROCESSOR_INFO[script_prefix][2] \
            if script_prefix in PROCESSOR_INFO else DEFAULT_PROCESSORS[2]
        self.tokenizer = tokenizer
        self.p_max_len = data_args.p_max_len
        self.proc_num = data_args.dataset_proc_num
        self.separator = getattr(self.tokenizer, data_args.passage_field_separator, data_args.passage_field_separator)

    def process(self, shard_num=1, shard_idx=0):
        self.dataset = self.dataset.shard(shard_num, shard_idx)
        if self.preprocessor is not None:
            self.dataset = self.dataset.map(
                self.preprocessor(self.tokenizer, self.p_max_len, self.separator),
                batched=False,
                num_proc=self.proc_num,
                remove_columns=self.dataset.column_names,
                desc="Running tokenization",
            )
        return self.dataset
