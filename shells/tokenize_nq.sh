## *************************************************
## Tokenize NQ Dataset
## *************************************************
DATA_DIR=/home/sunsi/dataset/nq
TOKENIZER=bert-base-uncased
TOKENIZER_ID=bert

## **********************************************
## train queries
## **********************************************
python ../preprocess/tokenize_nq_triviaqa_queries.py \
--tokenizer_name ${TOKENIZER} \
--query_file ${DATA_DIR}/nq-train-qrels.jsonl \
--save_to ${DATA_DIR}/${TOKENIZER_ID}/query/train.query.json \

# **********************************************
# test queries
# **********************************************
python ../preprocess/tokenize_nq_triviaqa_queries.py \
--tokenizer_name ${TOKENIZER} \
--query_file ${DATA_DIR}/nq-test.jsonl \
--save_to ${DATA_DIR}/${TOKENIZER_ID}/query/test.query.json \