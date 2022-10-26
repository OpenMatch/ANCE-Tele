import logging
import os
import sys

from transformers import AutoConfig, AutoTokenizer
from transformers import (
    HfArgumentParser,
    set_seed,
)
from transformers.integrations import TensorBoardCallback

# ## ------- Modified by SS.
# import sys
# sys.path.append("..")
# sys.path.append(os.getcwd()) ## cloud
# ## ------- Modified by SS.

from tevatron.arguments import ModelArguments, DataArguments
from tevatron.arguments import DenseTrainingArguments as TrainingArguments
from tevatron import trainers
from tevatron import utils
from tevatron import networks
from tevatron import dataloaders

logger = logging.getLogger(__name__)

def deactivate_relevant_gradients(model, trainable_components=['bias']):
    """Turns off the model parameters requires_grad except the trainable_components.
    Args:
        trainable_components (List[str]): list of trainable components (the rest will be deactivated)
    """
    i = 0
    num = 0
    for param in model.parameters():
        param.requires_grad = False
        num += 1
    if trainable_components:
        trainable_components = trainable_components + ['pooler.dense.bias']
    for name, param in model.named_parameters():
        for component in trainable_components:
            if component in name:
                param.requires_grad = True
                i += 1
                break
    print("only tune %d params took %.2f percent"%(i, (i/num)*100))

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
        do_train=True,
    )
    
    if model_args.bitfit:
        print("setting bitfit ...")
        deactivate_relevant_gradients(model)

    ## Train dataset and batchfy
    train_dataset, eval_dataset, QPCollator = dataloaders.get_train_dataset(
        tokenizer=tokenizer, 
        data_args=data_args,
        is_cross_encoder=model_args.cross_encoder,
    )
    
    ## tensorboard
    callbacks = None
    if training_args.tensorboard:
        callbacks = [TensorBoardCallback()]
    
    ## training func
    trainer = trainers.get_trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=utils.compute_reranking if model_args.cross_encoder else None,
        data_collator=QPCollator(
            tokenizer,
            max_p_len=data_args.p_max_len,
            max_q_len=data_args.q_max_len
        ),
        is_cross_encoder=model_args.cross_encoder,
        callbacks=callbacks,
    )
            
    train_dataset.trainer = trainer

    trainer.train()  # TODO: resume training

    trainer.save_model()
    if trainer.is_world_process_zero():
        tokenizer.save_pretrained(training_args.output_dir)


if __name__ == "__main__":
    main()
