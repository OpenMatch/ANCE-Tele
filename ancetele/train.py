import logging
import os
import sys

from transformers import AutoConfig, AutoTokenizer
from transformers import (
    HfArgumentParser,
    set_seed,
)
from transformers.integrations import TrainerCallback, TensorBoardCallback

# ## ------- Modified by SS.
# import sys
# sys.path.append("..")
# sys.path.append(os.getcwd()) ## cloud
# ## ------- Modified by SS.

# from arguments import ModelArguments, DataArguments
# from arguments import DenseTrainingArguments as TrainingArguments
# from ancetele import trainers
# from ancetele import utils
# from ancetele import networks
# from ancetele import dataloaders

from arguments import ModelArguments, DataArguments
from arguments import DenseTrainingArguments as TrainingArguments
import trainers
import networks
import dataloaders


logger = logging.getLogger(__name__)


class MyStopTrainCallback(TrainerCallback):
    "A callback that prints a message at the end of training step"

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step == args.early_stop_step:
            logger.info("End training at step: %d", state.global_step)
            control.should_training_stop = True
            
        return control
        

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

    ## Model
    model = networks.get_network(
        model_args,
        data_args,
        training_args,
        config=config,
        cache_dir=model_args.cache_dir,
        do_train=True,
    )

    ## Train dataset and batchfy
    train_dataset, eval_dataset, QPCollator = dataloaders.get_train_dataset(
        tokenizer=tokenizer, 
        data_args=data_args,
    )
    
    ## early-stop or tensorboard
    callbacks = []
    if training_args.early_stop_step > 0:
        logger.info("Setting early stop step at: %d", training_args.early_stop_step)
        callbacks.append(MyStopTrainCallback)
    if training_args.tensorboard:
        logger.info("Setting Tensorboard ...")
        callbacks.append(TensorBoardCallback())
    
    ## training func
    trainer = trainers.get_trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=QPCollator(
            tokenizer,
            max_p_len=data_args.p_max_len,
            max_q_len=data_args.q_max_len
        ),
        callbacks=callbacks,
    )
            
    train_dataset.trainer = trainer

    trainer.train()  # TODO: resume training

    trainer.save_model()
    if trainer.is_world_process_zero():
        tokenizer.save_pretrained(training_args.output_dir)


if __name__ == "__main__":
    main()
