import os
import sys
import copy
import json
import logging
import numpy as np
from tqdm import tqdm
from datasets import load_dataset
from collections import defaultdict
from transformers import AutoConfig, AutoTokenizer
from transformers import (
    HfArgumentParser,
    set_seed,
)
from transformers.integrations import TrainerCallback, TensorBoardCallback
from arguments import ModelArguments, DataArguments
from arguments import DenseTrainingArguments as TrainingArguments

import trainers
import networks
import dataloaders

logger = logging.getLogger(__name__)


def load_train_dataset(train_dir):
    train_files = [
        os.path.join(train_dir, f)
        for f in os.listdir(train_dir)
        if f.endswith('jsonl') or f.endswith('json')
    ]
    
    train_dataset = load_dataset(
        "json",
        "default",
        data_files={"train": train_files}, 
        cache_dir=os.path.join(train_dir, "cache") ,
    )["train"]
    return train_dataset


def get_full_split_dataset(file_path, mode="train"):
    dataset = json.load(open(file_path))
    data_dict = defaultdict(list)
    # name: base/novel
    # _class: p31. p32, ...
    # mode: train/test
    # qid-num, ......
    # p31 classe: 12, 14, 16, ....
    for name in dataset.keys():
        for _class in dataset[name].keys():
            data_dict[_class].extend(dataset[name][_class][mode])
    return data_dict


def sampling_fewshot_fct(data_dict, n_shot):
    selected_qidx_list = []
    for _class in data_dict.keys():
        
        selected_qidx_list.extend(
            np.random.choice(a=data_dict[_class], size=n_shot, replace=False, p=None)
        )
    assert len(set(selected_qidx_list)) == len(selected_qidx_list)
    print("tot few-shot examples: ", len(selected_qidx_list))
    return selected_qidx_list

def freeze_encoder_fct(model, freeze_encoder_name):
    i = 0
    num = 0
    for name, param in model.named_parameters():
        num += 1
        if freeze_encoder_name in name:
            param.requires_grad = False
            i += 1
    print("*************************************")
    print("Freeze {} encoder!".format(freeze_encoder_name))
    print("only tune %d params took %.2f percent"%(i, (i/num)*100))
    print("*************************************")
    


## *********************************************
## *********************************************
def main():
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))

    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()
        model_args: ModelArguments
        data_args: DataArguments
        training_args: TrainingArguments

    if (
            os.path.exists(training_args.output_dir)
            and os.listdir(training_args.output_dir)
            and training_args.do_train
            and not training_args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({training_args.output_dir}) \
            already exists and is not empty. Use --overwrite_output_dir to overcome."
        )

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
    )
    
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        training_args.local_rank,
        training_args.device,
        training_args.n_gpu,
        bool(training_args.local_rank != -1),
        training_args.fp16,
    )
    logger.info("Training/evaluation parameters %s", training_args)
    logger.info("MODEL parameters %s", model_args)

    set_seed(training_args.seed)
    
    ## *********************************
    print("*****************************************")
    print("random seed: ", training_args.seed)
    print("*****************************************")
    
    
    ## *********************************
    ## Model Setup
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

    ## Model
    model = networks.get_network(
        model_args,
        data_args,
        training_args,
        config=config,
        tokenizer=tokenizer,
        cache_dir=model_args.cache_dir,
        do_train=True,
    )
    
    ## *****************************************
    ## Whether freeze qry/psg-encoder
    if training_args.freeze_encoder_name:
        assert model_args.untie_encoder
        freeze_encoder_fct(model, training_args.freeze_encoder_name)
        
    ## *****************************************
    ## deep copy starting model
    original_state = copy.deepcopy(model.state_dict())
    original_output_dir=training_args.output_dir
    
    ## *****************************************
    ## load split dataset
    split_data_dict = get_full_split_dataset(file_path=data_args.split_dataset_stg)
    ## load dataset (base+novel)
    full_dataset = load_train_dataset(train_dir=data_args.train_dir)
    print("tot classes: {}, tot examples: {}".format(len(split_data_dict), len(full_dataset)))
    
    ## ***********************************
    ## tensorboard
    callbacks = []
    if training_args.tensorboard:
        logger.info("Setting Tensorboard ...")
        callbacks.append(TensorBoardCallback())
        
    ## *****************************************
    for n_shot in training_args.fewshot_extends:
        ## random select qidx
        few_shot_qidx_list = sampling_fewshot_fct(split_data_dict, n_shot=n_shot)

        ## generate few-shot dataset
        fewshot_dataset = full_dataset.filter(
            lambda data: data["qid-num"] in few_shot_qidx_list, load_from_cache_file=False) 
        ## load_from_cache_file must be False!
        fewshot_dataset.shard(num_shards=1, index=0)

        ## few-shot dataloader
        fewshot_train_dataset = dataloaders.DenseTrainDataset(data_args, dataset=fewshot_dataset, tokenizer=tokenizer)

        ## *********************************
        print("*****************************************")
        print("{}-shot/seed-{} ...".format(n_shot, training_args.seed))
        print("tot few-shot examples: ", 
              len(fewshot_train_dataset.train_data["qid-num"]))
        print("sampled few-shot qidxs: ", fewshot_train_dataset.train_data["qid-num"])
        print("*****************************************")

        ## *********************************
        ## recover original model
        model.load_state_dict(original_state)
        if model_args.param_efficient:
            model_class = networks.get_delta_model_class(model_args.param_efficient)
            delta_model = model_class(model)
            delta_model.freeze_module(set_state_dict=True)
            logger.info("Using param efficient method: %s", model_args.param_efficient)
        ## adjust output dir
        training_args.output_dir = os.path.join(original_output_dir, "{}-shot_seed-{}".format(n_shot, training_args.seed))
        training_args.run_name = original_output_dir.split("/")[-1] + "_{}-shot_seed-{}".format(n_shot, training_args.seed)

        ## training func
        trainer = trainers.get_trainer(
            model=model,
            args=training_args,
            train_dataset=fewshot_train_dataset,
            eval_dataset=None,
            data_collator=dataloaders.DenseQPCollator(
                tokenizer,
                max_p_len=data_args.p_max_len,
                max_q_len=data_args.q_max_len
            ),
            callbacks=callbacks,
            delta_model=delta_model if model_args.param_efficient else None
        )

        fewshot_train_dataset.trainer = trainer
        trainer.train()  # TODO: resume training
        trainer.save_model()

        if trainer.is_world_process_zero():
            tokenizer.save_pretrained(training_args.output_dir)

## *********************************************
## *********************************************
if __name__ == "__main__":
    main()
