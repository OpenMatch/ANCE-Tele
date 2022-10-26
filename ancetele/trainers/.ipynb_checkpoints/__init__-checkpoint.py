from .cross_trainer import CrossTrainer
from .dense_trainer import DenseTrainer, GCDenseTrainer


def get_trainer(
    model,
    args,
    train_dataset,
    eval_dataset,
    compute_metrics,
    data_collator,
    is_cross_encoder,
    callbacks=None,
):
    if is_cross_encoder:
        return CrossTrainer(
            model=model,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            compute_metrics=compute_metrics,
            data_collator=data_collator,
            callbacks=callbacks,
        )
    else:
        
        if args.grad_cache:            
            return GCDenseTrainer(
                model=model,
                args=args,
                train_dataset=train_dataset,
                data_collator=data_collator,
                callbacks=callbacks,
            )
            
        else:        
            return DenseTrainer(
                model=model,
                args=args,
                train_dataset=train_dataset,
                data_collator=data_collator,
                callbacks=callbacks,
            )