## *************************************************
## Tokenize Wikipedia Corpus
## *************************************************
DATA_DIR=/home/sunsi/dataset/wikipedia-corpus-index
TOKENIZER=bert-base-uncased
TOKENIZER_ID=bert

## **********************************************
## Corpus
## **********************************************
python ../preprocess/tokenize_wikipedia_passages.py \
--tokenizer_name ${TOKENIZER} \
--file ${DATA_DIR}/psgs_w100.tsv \
--save_to ${DATA_DIR}/${TOKENIZER_ID}/corpus \
--n_splits 20 \