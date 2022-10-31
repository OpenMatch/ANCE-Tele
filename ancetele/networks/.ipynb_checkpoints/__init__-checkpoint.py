import sys 
sys.path.append("..")
from arguments import ModelArguments, DataArguments
from arguments import DenseTrainingArguments as TrainingArguments
from .DenseRetriever import (DenseModel, DenseModelForInference)
from collections import OrderedDict

def get_network(
    model_args: ModelArguments,
    data_args: DataArguments,
    training_args: TrainingArguments,
    config: OrderedDict,
    cache_dir: str,
    do_train: bool,
):
    if do_train:
        model = DenseModel.build(
            model_args,
            data_args,
            training_args,
            config=config,
            cache_dir=model_args.cache_dir,
        )
    else:
        model = DenseModelForInference.build(
            model_name_or_path=model_args.model_name_or_path,
            data_args=data_args,
            config=config,
            cache_dir=model_args.cache_dir,
        )

    return model
    
