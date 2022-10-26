def compute_reranking_minus_loss(EvalPrediction):
    minus_loss = sum(EvalPrediction.predictions) / len(EvalPrediction.predictions)
    return {"eval_minus_loss": minus_loss}

# def compute_reranking_minus_loss(EvalPrediction):
#     minus_loss = - sum(EvalPrediction.predictions)
#     return {"eval_minus_loss": minus_loss}