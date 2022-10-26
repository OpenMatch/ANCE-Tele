## *************************************************
## Tokenize
## *************************************************
DATA_DIR=/data/private/sunsi/dataset/msmarco/rocketqa
TOKENIZER=bert-base-uncased
TOKENIZER_ID=bert

## train queries
python ../preprocess/tokenize_queries.py \
--tokenizer_name ${TOKENIZER} \
--query_file ${DATA_DIR}/train.query.txt \
--save_to ${DATA_DIR}/${TOKENIZER_ID}/query/train.query.json \

## dev queries
python ../preprocess/tokenize_queries.py \
--tokenizer_name ${TOKENIZER} \
--query_file ${DATA_DIR}/dev.query.txt \
--save_to ${DATA_DIR}/${TOKENIZER_ID}/query/dev.query.json \

## corpus
python ../preprocess/tokenize_passages.py \
--tokenizer_name ${TOKENIZER} \
--file ${DATA_DIR}/corpus.tsv \
--save_to ${DATA_DIR}/${TOKENIZER_ID}/corpus \


## train-positives
python ../preprocess/tokenize_train_positives.py \
--data_dir ${DATA_DIR} \
--tokenizer_name ${TOKENIZER} \
--save_to ${DATA_DIR}/${TOKENIZER_ID}/query/train.positives.json \