export DATA_DIR=/home/sunsi/dataset/nq
export OUTPUT_DIR=/home/sunsi/experiments/nq-results
export CORPUS_DATA_DIR=/home/sunsi/dataset/wikipedia-corpus-index
## *************************************
## INPUT/OUTPUT
export train_job_name=epi-1.ance-tele.nq.checkp-2000
export infer_job_name=inference.${train_job_name}
## OUTPUT
export new_ann_hn_file_name=ann-neg.${train_job_name}
export new_la_hn_file_name=la-neg.${train_job_name}
export new_tele_file_name_wo_mom=ann-la-neg.${train_job_name}

export mom_tele_file_name=epi-1-tele-neg.nq
export new_tele_file_name=epi-2-tele-neg.nq
## *************************************

## *************************************
## ENCODE Corpus GPUs
ENCODE_CUDA="0,1,2,3,4" ## ENCODE_CUDA="0"
ENCODE_CUDAs=(${ENCODE_CUDA//,/ })
ENCODE_CUDA_NUM=${#ENCODE_CUDAs[@]}
## Search Top-k GPUs
SEARCH_CUDA="0,1,2,3,4"
## *************************************
## Length SetUp
export q_max_len=32
export p_max_len=156
## *************************************
TOKENIZER=bert-base-uncased
TOKENIZER_ID=bert
SplitNum=20 ## Wikipedia is splited into 20 sub-files
## *************************************

## **********************************************
## Infer
## **********************************************
## Create Folder
mkdir -p ${OUTPUT_DIR}/${infer_job_name}/corpus
mkdir -p ${OUTPUT_DIR}/${infer_job_name}/query

## Encoding Corpus
for((tmp=0; tmp<$SplitNum; tmp+=$ENCODE_CUDA_NUM))
do
    ## *************************************
    for((CUDA_INDEX=0; CUDA_INDEX<$ENCODE_CUDA_NUM; CUDA_INDEX++))
    do
        ## *************************************
        if [ $[CUDA_INDEX + $tmp] -eq $SplitNum ]
        then
          break 2
        fi

        ## *************************************
        printf -v i "%02g" $[CUDA_INDEX + $tmp] &&
        CUDA=${ENCODE_CUDAs[$CUDA_INDEX]} &&
        echo ${OUTPUT_DIR}/${train_job_name} &&
        echo split-${i} on gpu-${CUDA} &&

        CUDA_VISIBLE_DEVICES=${CUDA} python ../ancetele/encode.py \
        --output_dir ${OUTPUT_DIR}/${infer_job_name} \
        --model_name_or_path ${OUTPUT_DIR}/${train_job_name}/passage_model \
        --fp16 \
        --per_device_eval_batch_size 1024 \
        --dataloader_num_workers 2 \
        --p_max_len ${p_max_len} \
        --encode_in_path ${CORPUS_DATA_DIR}/${TOKENIZER_ID}/corpus/split${i}.json \
        --encoded_save_path ${OUTPUT_DIR}/${infer_job_name}/corpus/split${i}.pt &> \
        ${OUTPUT_DIR}/${infer_job_name}/corpus/split${i}.log &&
        ## *************************************
        sleep 3 &
        [ $CUDA_INDEX -eq `expr $ENCODE_CUDA_NUM - 1` ] && wait
    done
done


## *************************************
## Encode [Train Query]
## *************************************
CUDA_VISIBLE_DEVICES=${ENCODE_CUDAs[-1]} python ../ancetele/encode.py \
--output_dir ${OUTPUT_DIR}/${infer_job_name} \
--model_name_or_path ${OUTPUT_DIR}/${train_job_name}/query_model \
--fp16 \
--q_max_len ${q_max_len} \
--encode_is_qry \
--per_device_eval_batch_size 1024 \
--encode_in_path ${DATA_DIR}/${TOKENIZER_ID}/query/train.query.json \
--encoded_save_path ${OUTPUT_DIR}/${infer_job_name}/query/train.query.pt \


## *************************************
## Search [Train]
## *************************************
CUDA_VISIBLE_DEVICES=${SEARCH_CUDA} python ../ancetele/faiss_retriever/do_retrieval.py \
--query_reps ${OUTPUT_DIR}/${infer_job_name}/query/train.query.pt \
--passage_reps ${OUTPUT_DIR}/${infer_job_name}/corpus/'*.pt' \
--index_num ${SplitNum} \
--batch_size 1024 \
--use_gpu \
--save_text \
--depth 200 \
--save_ranking_to ${OUTPUT_DIR}/${infer_job_name}/train.rank.tsv \
--sub_split_num 5 \
## if CUDA memory is not enough, set this augment.


# # ***************************************************
# # Filter [Train] & Generate ANN Negatives & Generate [Train-Positive]
# # ***************************************************
python ../preprocess/build_train_em_hn.py \
--tokenizer_name ${TOKENIZER} \
--input_file ${OUTPUT_DIR}/${infer_job_name}/train.rank.tsv \
--queries ${DATA_DIR}/nq-train-qrels.jsonl \
--collection ${CORPUS_DATA_DIR}/psgs_w100.tsv \
--save_to ${DATA_DIR}/${TOKENIZER_ID}/${new_ann_hn_file_name} \
--n_sample 80 \
--depth 200 \
--gen_pos_file ${OUTPUT_DIR}/${infer_job_name}/train.positives.json \
--mark hn \


# # ***************************************************
# # Encode [Train-Positive]
# # ***************************************************
CUDA_VISIBLE_DEVICES=${ENCODE_CUDAs[-1]} python ../ancetele/encode.py \
--output_dir ${OUTPUT_DIR}/${infer_job_name} \
--model_name_or_path ${OUTPUT_DIR}/${train_job_name}/passage_model \
--fp16 \
--p_max_len ${p_max_len} \
--per_device_eval_batch_size 1024 \
--encode_in_path ${OUTPUT_DIR}/${infer_job_name}/train.positives.json \
--encoded_save_path ${OUTPUT_DIR}/${infer_job_name}/query/train.positives.pt \


## ***************************************************
## Search [Train-Positive]
## ***************************************************
CUDA_VISIBLE_DEVICES=${SEARCH_CUDA} python ../ancetele/faiss_retriever/do_retrieval.py \
--query_reps ${OUTPUT_DIR}/${infer_job_name}/query/train.positives.pt \
--passage_reps ${OUTPUT_DIR}/${infer_job_name}/corpus/'*.pt' \
--index_num ${SplitNum} \
--batch_size 1024 \
--use_gpu \
--save_text \
--depth 200 \
--save_ranking_to ${OUTPUT_DIR}/${infer_job_name}/train.positives.rank.tsv \
--sub_split_num 5 \
## if CUDA memory is not enough, set this augment.

# ***************************************************
# Filter [Train-Positive] & Generate LA Negatives
# ***************************************************
python ../preprocess/build_train_em_hn.py \
--tokenizer_name ${TOKENIZER} \
--input_file ${OUTPUT_DIR}/${infer_job_name}/train.positives.rank.tsv \
--queries ${DATA_DIR}/nq-train-qrels.jsonl \
--collection ${CORPUS_DATA_DIR}/psgs_w100.tsv \
--save_to ${DATA_DIR}/${TOKENIZER_ID}/${new_la_hn_file_name} \
--n_sample 80 \
--depth 200 \
--mark la.hn \


# # *************************************
# # Combine ANN + LA Negatives
# # *************************************
python ../preprocess/combine_nq_triviaqa_negative.py \
--data_dir ${DATA_DIR}/${TOKENIZER_ID} \
--input_folder_1 ${new_la_hn_file_name} \
--input_folder_2 ${new_ann_hn_file_name} \
--output_folder ${new_tele_file_name_wo_mom} \


# # *************************************
# #  Combine (ANN + LA + Mom) Negatives
# # *************************************
python ../preprocess/combine_nq_triviaqa_negative.py \
--data_dir ${DATA_DIR}/${TOKENIZER_ID} \
--input_folder_1 ${mom_tele_file_name} \
--input_folder_2 ${new_tele_file_name_wo_mom} \
--output_folder ${new_tele_file_name} \