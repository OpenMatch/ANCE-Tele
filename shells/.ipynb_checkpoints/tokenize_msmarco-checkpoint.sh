## *************************************************
## Tokenize MS MARCO Dataset
## *************************************************
DATA_DIR=/home/sunsi/dataset/msmarco/msmarco
TOKENIZER=bert-base-uncased
TOKENIZER_ID=bert

# ## corpus
# python ../preprocess/tokenize_marco_passages.py \
# --tokenizer_name ${TOKENIZER} \
# --file ${DATA_DIR}/corpus.tsv \
# --save_to ${DATA_DIR}/${TOKENIZER_ID}/corpus \

# ## train queries
# python ../preprocess/tokenize_marco_queries.py \
# --tokenizer_name ${TOKENIZER} \
# --query_file ${DATA_DIR}/train.query.txt \
# --save_to ${DATA_DIR}/${TOKENIZER_ID}/query/train.query.json \

## train positives
python ../preprocess/tokenize_marco_positives.py \
--data_dir ${DATA_DIR} \
--tokenizer_name ${TOKENIZER} \
--save_to ${DATA_DIR}/${TOKENIZER_ID}/query/train.positives.json \

# ## dev queries
# python ../preprocess/tokenize_marco_queries.py \
# --tokenizer_name ${TOKENIZER} \
# --query_file ${DATA_DIR}/dev.query.txt \
# --save_to ${DATA_DIR}/${TOKENIZER_ID}/query/dev.query.json