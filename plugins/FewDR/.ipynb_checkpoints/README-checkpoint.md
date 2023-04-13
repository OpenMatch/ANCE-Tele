# FewDR

## Outline

- [FewDR](#fewdr)
  - [Outline](#outline)
  - [Overview](#overview)
  - [Requirements](#requirements)
  - [Reproduce Our Results](#reproduce-our-results)
  - [Contact Us](#contact-us)


## Overview

The results of FewDR benchmark can be found in [result.csv](result.csv).

## Requirements

FewDR is tested on Python 3.8+, PyTorch 1.8+, and CUDA Version 10.2/11.1.

(1) Create a new Anaconda environment:

```
conda create --name fewdr python=3.8
conda activate fewdr
```

(2) Install the following packages using Pip or Conda under this environment:
```
transformers==4.9.2
datasets==1.11.0
pytorch==1.8.0

faiss-gpu==1.7.2
## faiss-gpu is depend on the CUDA version
## conda install faiss-gpu cudatoolkit=11.1
## conda install faiss-gpu cudatoolkit=10.2

openjdk==11
pyserini ## pyserini is depend on openjdk
```

## Reproduce Our Results

### FewDR Download

### Data Preprocess

Finally, the format of the tokenized data is as follows:

```
{
  "qid-num": "query id number in string format",
  "query": [train-query tokenized ids],
  "positives": [[positive-passage-1 tokenized ids], [positive-passage-2 tokenized ids], ...],
  "negatives": [[negative-passage-1 tokenized ids], [negative-passage-2 tokenized ids], ...],
}
```

Here is a toy of tokenized FewDR base data: [base-train-toy.json](data/base-train-toy.json). The whole benchmark will be released in the future.

### Model Training

After data preprocessing, you can enter the folder `FewDR/shells` and run the corresponding shell script to train your model.

For zero-shot train, use only our base-train data as train data and run following command:

```shell
bash zero-shot-train.sh
```

For full-shot train, use both base-train and novel-train data and run following command:

```shell
bash full-shot-train.sh
```

For few-shot train, set your zero-shot trained model as pretrained model,  use both base-train and novel-train data and run following command:

```shell
bash few-shot-train.sh
```

Different few-shot seeds can be set in the `few-shot-train.sh` shell script.

P.S. Multi-GPU training is supported. Please keep the following hyperparameters unchanged and set `--negatives_x_device` when using multi-GPU setup.

| Hyperparameters                        | Augments                      | Single GPU | E.g., Two GPUs |
| :------------------------------------- | :---------------------------- | :--------- | :------------- |
| Qry Batch Size                         | --per_device_train_batch_size | 32         | 16             |
| (Positive + Negative) Passages per Qry | --train_n_passages            | 2          | 2              |
| Learning rate                          | --learning_rate               | 5e-6       | 5e-6           |
| Total training Epoch                   | --num_train_epochs            | 40         | 40             |



#### Grad Cache Notice

> 🙌 If your CUDA memory is limited, please use the [Gradient Caching](https://arxiv.org/pdf/2101.06983.pdf) technique. Set the following augments during training:
```
--grad_cache \
--gc_q_chunk_size 4 \
--gc_p_chunk_size 8 \
## Split a batch of queries to several gc_q_chunk_size
## Split a batch of passages to several gc_p_chunk_size
```



### Inference and Evaluation

After model training, you can use our shell scripts to build inference and make evaluation for your checkpoints.

#### Process data

The format of test query and corpus used in our inference shell script is as follows:

```
{
  "text_id": "text id in string format",
  "text": [text tokenized ids],
}
```

We support multi-GPUs to encode the Wiki corpus, which is split into twenty files (split00-split19).

The Wikipedia-Corpus-Index can be downloaded here:

| Download Link                                                | Size   |
| :----------------------------------------------------------- | :----- |
| [wikipedia-corpus-index.tar.gz](https://thunlp.oss-cn-qingdao.aliyuncs.com/PaperData/EMNLP2022/ANCE-Tele/wikipedia-corpus-index.tar.gz) | ~12.9G |

Run the command: `tar -zxvf wikipedia-corpus-index.tar.gz`. The uncompressed folder contains the following files:

```
wikipedia-corpus-index

  ├── psgs_w100.tsv # <TSV> psg_id /t psg /t psg_title

  └── index-wikipedia-dpr-20210120-d1b9e6 # Wikipedia Index (for pyserini evaluation)
```

#### Build inference and make evaluation

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
