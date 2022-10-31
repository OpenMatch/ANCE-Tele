# ANCE-Tele

This is the implementation of ANCE-Tele introduced in the EMNLP 2022 Main Conference paper **"Reduce Catastrophic Forgetting of Dense Retrieval Training with Teleportation Negatives"**. If you find this work useful, please cite our paper 😃 and give our repo a star ⭐️. Thanks♪(･ω･)ﾉ

```
@inproceedings{sun2022ancetele,
  title={Reduce Catastrophic Forgetting of Dense Retrieval Training with Teleportation Negatives},
  author={Si Sun, Chenyan Xiong, Yue Yu, Arnold Overwijk, Zhiyuan Liu and Jie Bao},
  booktitle={Proceedings of EMNLP 2022},
  year={2022}
}
```
## Outline

- [ANCE-Tele](#ance-tele)
  - [Overview](#overview)
  - [Requirements](#requirements)
  - [Reproduce MS MARCO Results](#reproduce-ms-marco-results)
  - [Reproduce NQ Results](#reproduce-nq-results) [Will be updated on 11/1]
  - [Reproduce TriviaQA Results](#reproduce-triviaqa-results) [Will be updated on 11/2]
  - [Easy-to-Use Tips](#easy-to-use-tips)
  - [Contact Us](#acknowledgement)
  - [Acknowledgement](#acknowledgement)


## Overview

<img src="framework.jpeg">

ANCE-Tele is a simple and efficient DR training method that introduces teleportation (momentum and lookahead) negatives to smooth the learning process, leading to ***improved training stability, convergence speed, and reduced catastrophic forgetting***.

On web search and OpenQA, ANCE-Tele is ***competitive*** among systems using significantly more (50x) parameters, and ***eliminates the dependency on additional negatives (e.g., BM25, other DR systems), filtering strategies, and distillation modules.*** You can easily reproduce ANCE-Tele about one day with only an A100 😉. (Of course, 2080Ti is ok but with more time).

Let's begin!


## Requirements

ANCE-Tele is tested on Python 3.8+, PyTorch 1.8+, and CUDA Version 10.2/11.1.

[1] Create a new Anaconda environment:

```
conda create --name ancetele python=3.8
conda activate ancetele
```

[2] Install the following packages using Pip or Conda under this environment:
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


## Reproduce MS MARCO Results

- [MS MARCO Quick Link](#reproduce-ms-marco-results)
  - [MARCO: Download](#marco-download)
  - [MARCO: Preprocess](#marco-preprocess)
  - [MARCO: Reproduce w/ Our CheckPs](#marco-reproduce-using-our-checkps)
  - [MARCO: Reproduce w/ Our Episode-3 Training Negatives](#marco-reproduce-using-our-episode-3-training-negatives)
  - [MARCO: Reproduce from Scratch (Episode 1->2->3)](#marco-reproduce-from-scratch)


### MARCO Download

[1] Download Dataset

|Download Link|Size|
|:-----|:----|
|[msmarco.tar.gz](https://thunlp.oss-cn-qingdao.aliyuncs.com/PaperData/EMNLP2022/ANCE-Tele/msmarco.tar.gz)|~1.0G|

[2] Uncompress Dataset

Run the command: `tar -zxcf msmarco.tar.gz`. The uncompressed folder contains the following files:

```
msmarco
  ├── corpus.tsv  # <FORMAT> psg_id /t psg_title /t psg
  ├── train.query.txt  # <FORMAT> qry_id /t qry
  ├── qrels.train.tsv  # <FORMAT> qry_id /t 0 /t pos_psg_id /t 1
  ├── dev.query.txt  # <FORMAT> qry_id /t qry
  └── qrels.dev.small.tsv  # <FORMAT> qry_id /t 0 /t pos_psg_id /t 1
```

### MARCO Preprocess

[1] Tokenize Dataset

Enter the folder `ANCE-Tele/shells` and run the shell script:
```
bash tokenize_msmarco.sh
```

### MARCO Reproduce using Our CheckPs

[1] Download our CheckP from HuggingFace:

|Download Link|Size|Dev MRR@10|
|:-----|:----|:----:|
|[ance-tele_msmarco_qry-psg-encoder](https://huggingface.co/OpenMatch/ance-tele_msmarco_qry-psg-encoder)|~438M|39.1|

[2] Encoder & Search MS MARCO using our CheckP:

Enter the folder `ANCE-Tele/shells` and run the shell script:
```
bash infer_msmarco.sh
```
P.S. We support multi-gpus to encode the MARCO corpus, which is split into 10 files (split00-split09). But Multi-gpu encoding only supports the use of 1/2/5 gpus at the same time, e.g., setting `ENCODE_CUDA="0,1"`

#### Faiss Notice

> Faiss-GPU search is also supported but requires sufficient CUDA memory. In our experience, CUDA memory >= 1\*A100 or 2\*3090 or 4\*2080ti is ok, e.g., setting `SEARCH_CUDA="0,1,2,3"`. If your CUDA memory is still not enough, you can use split search: set `--sub_split_num 5` and the *sub_split_num* can be 1/2/5

> You can of course also use Faiss-CPU search: (1) do not use `--use_gpu` and set `--batch_size -1`.


### MARCO Reproduce using Our Episode-3 Training Negatives

[1] Download vanilla pre-trained model & our Epi-3 training negatives:

|Download Link|Size|
|:-----|:----|
|[co-condenser-marco](https://huggingface.co/Luyu/co-condenser-marco)|~473M|
|[ance-tele_msmarco_tokenized-train-data.tar.gz](https://thunlp.oss-cn-qingdao.aliyuncs.com/PaperData/EMNLP2022/ANCE-Tele/ance-tele_msmarco_tokenized-train-data.tar.gz)|~8.8G|

[2] Uncompress our Epi-3 training negatives:

Run the command: `tar -zxcf ance-tele_msmarco_tokenized-train-data.tar.gz`. The uncompressed folder contains 12 sub-files {split00-11.hn.json}. The format of each file is as follows:
```
{
  "query": [train-query tokenized ids],
  "positives": [[positive-passage-1 tokenized ids], [positive-passage-2 tokenized ids], ...],
  "negatives": [[negative-passage-1 tokenized ids], [negative-passage-2 tokenized ids], ...],
}
```

[3] Train ANCE-Tele using our Epi-3 training negtatives

Enter the folder `ANCE-Tele/shells` and run the shell script:
```
bash train_ance-tele_msmarco.sh
```
P.S. Multi-GPU training is supported. Please keep the following hyperparameters unchanged and set `--negatives_x_device` when using multi-GPU setup.

|Hyperparameters|Augments|Single GPU|E.g., Two GPUs|
|:-----|:-----|:-----|:-----|
|Qry Batch Size|--per_device_train_batch_size|8|4|
|(Positive + Negative) Passages per Qry|--train_n_passages|32|32|
|Learning rate|--learning_rate|5e-6|5e-6|
|Total training Epoch|--num_train_epochs|3|3|

#### Grad Cache Notice

> If your CUDA memory is limited, please use [Gradient Caching](https://arxiv.org/pdf/2101.06983.pdf) technique. Set the following augments during training:
```
--grad_cache \
--gc_q_chunk_size 4 \
--gc_p_chunk_size 8 \
## Split a batch of queries to several gc_q_chunk_size
## Split a batch of passages to several gc_p_chunk_size
```


[4] Evaluate your ANCE-Tele

After training for 3 epochs, you can follow the instructions in [MARCO: Reproduce w/ Our CheckPs](#marco-reproduce-using-our-checkps) to evaluate. Remember to replace the CheckP with your trained model file 😉.



### MARCO Reproduce from Scratch

If you want to reproduce ANCE-Tele from scratch (Epi->2->3), you just need to prepare the vanilla pretrained model [co-condenser-marco](https://huggingface.co/Luyu/co-condenser-marco).


#### Iterative Training Notice

> ANCE-Tele takes a quick refreshing strategy for hard negative mining. Hence, Epi-1,2 just train 1/10 of the total number of training steps (early stop), only the last Epi-3 go through the full training epoch, which greatly reduces the training cost.

> Every episode also adopts **train from scratch** mode, that is, each episode uses the vanilla pretrained model as the initial model, and the only difference is the training negatives. In this way, only the final training negatives are required to reproduce the results without relying on the intermediate CheckPs.


[1] Epi-1

First mine the Tele-negatives using the the vanilla *co-condenser-marco*. In Epi-1 , Tele-negatives contain ANN-negatives and Lookahead-negatives (LA-Neg) without Momentum.

```
bash epi-1-mine-msmarco.sh
```
Then train the vanilla *co-condenser-marco* with the Epi-1 Tele-negatives and early stop at 20k step for negative refreshing:
```
bash epi-1-train-msmarco.sh
```

[2] Epi-2

For Epi-2, mine Tele-negatives using the Epi-1 trained model. Epi-2 Tele-negatives contain ANN-negatives, Lookahead-negatives (LA-Neg), and Momentum-negatives (Epi-1 training negatives).

```
bash epi-2-mine-msmarco.sh
```
Then train the vanilla *co-condenser-marco* with the Epi-2 Tele-negatives and early stop at 20k step for negative refreshing:
```
bash epi-2-train-msmarco.sh
```

[3] Epi-3

For last Epi-3, mine Tele-negatives using the Epi-2 trained model. Epi-3 Tele-negatives contain ANN-negatives, Lookahead-negatives (LA-Neg), and Momentum-negatives (Epi-2 training negatives).

```
bash epi-3-mine-msmarco.sh
```
Then train the vanilla *co-condenser-marco* with the Epi-3 Tele-negatives. This step is the same as introduced in [MARCO: Reproduce w/ Our Episode-3 Training Negatives](#marco-reproduce-using-our-episode-3-training-negatives):
```
bash epi-3-train-msmarco.sh
```

[4] Evaluate your ANCE-Tele

After three episode, you can follow the instructions in [MARCO: Reproduce w/ Our CheckPs](#marco-reproduce-using-our-checkps) to evaluate. Remember to replace the CheckP with your trained model file 😉.


## Easy-to-Use Tips

* [Faiss Notice](#faiss-notice): Multi-GPU/CPU Search
* [Grad Cache Notice](#grad-cache-notice): Save CUDA Memory (Train ANCE-Tele with 2080ti)
* [Iterative Training Notice](#iterative-training-notice): ANCE-Tele takes a *quick-refreshing-strategy* and *train-from-scratch* mode.


## Contact Us

For any question, feel free to create an issue, and we will try our best to solve. If the problem is more urgent, you can send an email to me at the same time 🤗.

```
NAME: Si Sun
EMAIL: s-sun17@mails.tsinghua.edu.cn
```

## Acknowledgement

ANCE-Tele is implemented and modified based on [Tevatron](https://github.com/texttron/tevatron). We thank the authors for their open-sourcing and excellent work. We have integrated ANCE-Tele into [OpenMatch](https://github.com/OpenMatch/OpenMatch/blob/master/docs/Openmatch-ANCE-Tele.md).
