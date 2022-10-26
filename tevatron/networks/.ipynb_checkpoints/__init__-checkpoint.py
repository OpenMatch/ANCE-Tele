from ..arguments import ModelArguments, DataArguments
from ..arguments import DenseTrainingArguments as TrainingArguments
from collections import OrderedDict

def get_network(
    model_args: ModelArguments,
    data_args: DataArguments,
    training_args: TrainingArguments,
    config: OrderedDict,
    cache_dir: str,
    is_cross_encoder: bool,
    do_train: bool,
):
    ## Cross-Encoder
    if is_cross_encoder:
        ## train
        if do_train:
            model = CrossModel.build(
                model_args,
                data_args,
                training_args,
                config=config,
                cache_dir=model_args.cache_dir,
            )
        ## eval
        else:
            model = CrossModelForInference.build(
                model_name_or_path=model_args.model_name_or_path,
                config=config,
                cache_dir=model_args.cache_dir,
            )
    ## Dual-Encoder  
    else:
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
    
from .CrossEncoder import (CrossModel, CrossModelForInference)
from .DenseRetriever import (DenseModel, DenseModelForInference)