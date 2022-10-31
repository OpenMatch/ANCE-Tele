from .dense_trainer import DenseTrainer, GCDenseTrainer


def get_trainer(
    model,
    args,
    train_dataset,
    eval_dataset,
    data_collator,
    callbacks=None,
):
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