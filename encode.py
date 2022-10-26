import logging
import os
import pickle
import sys
from contextlib import nullcontext

import numpy as np
from tqdm import tqdm

import json
import torch

from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoTokenizer
from transformers import (
    HfArgumentParser,
)

from tevatron import utils
from tevatron import networks
from tevatron import dataloaders

from tevatron.arguments import ModelArguments, DataArguments, \
    DenseTrainingArguments as TrainingArguments


logger = logging.getLogger(__name__)


def main():
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()
        model_args: ModelArguments
        data_args: DataArguments
        training_args: TrainingArguments

    if training_args.local_rank > 0 or training_args.n_gpu > 1:
        raise NotImplementedError('Multi-GPU encoding is not supported.')

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
    )

    num_labels = 1
    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        num_labels=num_labels,
        cache_dir=model_args.cache_dir,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=False,
    )
    
    ## Model (dense or cross)
    model = networks.get_network(
        model_args,
        data_args,
        training_args,
        config=config,
        cache_dir=model_args.cache_dir,
        is_cross_encoder=model_args.cross_encoder,
        do_train=False,
    )
    
    ## Train dataset and batchfy
    encode_dataset, EncodeCollator = dataloaders.get_encode_dataset(
        tokenizer=tokenizer, 
        data_args=data_args,
        is_cross_encoder=model_args.cross_encoder,
    )
    
    if model_args.cross_encoder:
        text_max_length = data_args.q_max_len + 1 + data_args.p_max_len
    else:
        text_max_length = data_args.q_max_len if data_args.encode_is_qry else data_args.p_max_len
    
    encode_loader = DataLoader(
        encode_dataset,
        batch_size=training_args.per_device_eval_batch_size,
        collate_fn=EncodeCollator(
            tokenizer,
            max_length=text_max_length,
            padding='max_length'
        ),
        shuffle=False,
        drop_last=False,
        num_workers=training_args.dataloader_num_workers,
    )

    model = model.to(training_args.device)
    model.eval()
    
    ## ***********************************
    ## Cross-Encoder
    ## ***********************************
    if model_args.cross_encoder:
        tot_scores = []
        for batch in tqdm(encode_loader):
            with torch.cuda.amp.autocast() if training_args.fp16 else nullcontext():
                with torch.no_grad():
                    for k, v in batch.items():
                        batch[k] = v.to(training_args.device)
                        
                    scores = model(query_passage=batch)
                    tot_scores.append(scores.cpu().detach().numpy())
                
        iter_scores = iter(np.concatenate(tot_scores))
        with open(os.path.join(data_args.encoded_save_path, 'train.json'), 'w') as fw:
            for ex in tqdm(encode_dataset.examples):
                qry = ex['query']
                positives = ex['positives']
                negatives = ex['negatives']
                
                pos_scores = []
                for _ in positives:
                    pos_scores.append(np.float(next(iter_scores)))

                neg_scores = []
                for _ in negatives:
                    neg_scores.append(np.float(next(iter_scores))) 
                    ## `np.float` is a deprecated alias for the builtin `float`.
                    
                item = {
                    'query': qry,
                    'positives': positives,
                    'negatives': negatives,
                    'pos_scores': pos_scores,
                    'neg_scores': neg_scores
                }                
                fw.write(json.dumps(item)+ '\n')
                
    ## ***********************************
    ## Dense-Encoder
    ## ***********************************  
    else:
        encoded = []
        lookup_indices = []
        for (batch_ids, batch) in tqdm(encode_loader):
            lookup_indices.extend(batch_ids)

            with torch.cuda.amp.autocast() if training_args.fp16 else nullcontext():
                with torch.no_grad():
                    for k, v in batch.items():
                        batch[k] = v.to(training_args.device)
                    if data_args.encode_is_qry:
                        model_output: DenseOutput = model(query=batch)
                        encoded.append(model_output.q_reps.cpu().detach().numpy())
                    else:
                        model_output: DenseOutput = model(passage=batch)
                        encoded.append(model_output.p_reps.cpu().detach().numpy())

        encoded = np.concatenate(encoded)

        with open(data_args.encoded_save_path, 'wb') as f:
            pickle.dump((encoded, lookup_indices), f)


if __name__ == "__main__":
    main()
