# FewDR

The repository is about our recent work **["Rethinking Few-shot Ability in Dense Retrieval"](https://arxiv.org/pdf/2304.05845.pdf)**. This is an ongoing project, and we will gradually open source our data and code here.


## Results

The results on FewDR benchmark can be found in [FewDR-Results.csv](./FewDR-Results.csv).

## Requirements

The installation requirements here are the same as those for [ANCE-Tele](../../README.md).


## FewDR Dataset

The statistics of the FewDR dataset are as follows:

|Split|Classes|Queries|#Train Qry|#Test Qry|Corpus Size|
|:---|:---|:---|:---|:---|:---|
|All| 60| 41,420| 20,726| 20,694| 21,015,324|
|Base| 30 |20,668 |10,341 |10,327 |21,015,324|
|Novel |30 |20,752 |10,385 |10,367|21,015,324|

One samples per line, formatted as follows:
```

"class": "P17",
"qid": "P17_97",
"qid-num": "24597",
"question": "which country is sharm el sheikh international airport located in",
"answers": ["egypt"],
"positive_ctxs": [
  {"title": "Sharm El Sheikh International Airport",
  "text": "Sharm El Sheikh International Airport is an international airport located in Sharm El Sheikh, Egypt. It is the third-busiest airport ...",
  "score": 1000,
  "passage_id": "9581501"}
  ],
}
```
Our data partition files are as follows:
```
{
  "base-train": [qid_btrain_1, qid_btrain_2, ...],
  "novel-train": [qid_btest_1, qid_btest_2, ...],
  "base-test": [qid_ntrain_1, qid_ntrain_2, ...],
  "novel-test": [qid_ntest_1, qid_ntest_2, ...],
}

```

P.S. The whole benchmark will be released in the future.

## Expriments


### Training


Before training, we should tokenize the training data. Here is a toy of tokenized FewDR base train data: [base-train-toy.json](data/base-train-toy.json). The format of the tokenized data is as follows:
```
{
  "qid-num": "query id number in string format",
  "query": [train-query tokenized ids],
  "positives": [[positive-passage-1 tokenized ids], [positive-passage-2 tokenized ids], ...],
  "negatives": [[negative-passage-1 tokenized ids], [negative-passage-2 tokenized ids], ...],
}
```

After data preprocessing, we can enter the folder `FewDR/shells` and run the corresponding shell script to train your model.


**For zero-shot train**, use only our base-train data as train data and run following command:

```shell
bash zero-shot-train.sh
```

**For full-shot train**, use both base-train and novel-train data and run following command:

```shell
bash full-shot-train.sh
```

**For few-shot train**, set your zero-shot trained model as pretrained model,  use both base-train and novel-train data and run following command:

```shell
bash few-shot-train.sh
```

**Different few-shot seeds can be set in the `few-shot-train.sh` shell script. Here we use 5 different seeds (41,42,43,44,45).**

Multi-GPU training is supported. Please keep the following hyperparameters unchanged and set `--negatives_x_device` when using multi-GPU setup.

| Hyperparameters                        | Augments                      | Single GPU | E.g., Two GPUs |
| :------------------------------------- | :---------------------------- | :--------- | :------------- |
| Qry Batch Size                         | --per_device_train_batch_size | 32         | 16             |
| (Positive + Negative) Passages per Qry | --train_n_passages            | 2          | 2              |
| Learning rate                          | --learning_rate               | 5e-6       | 5e-6           |
| Total training Epoch                   | --num_train_epochs            | 40         | 40             |


P.S. For more training & inference techniques, please see ANCE-Tele/README.md



### Inference and Evaluation

After model training, we can use our shell scripts to build inference and make evaluation for our checkpoints.
The format of test query and corpus used in our inference shell script is the same as ANCE-Tele (NQ):


(1) Zero-shot and Full-shot inference:

```shell
bash zero-full-shot-inference.sh
```

(2) Few-shot inference:

```shell
bash few-shot-inference.sh
```


## Contact Us

For any question, feel free to create an issue, and we will try our best to solve. If the problem is more urgent, you can send an email to me at the same time 🤗.

```
NAME: Si Sun
EMAIL: s-sun17@mails.tsinghua.edu.cn
```
