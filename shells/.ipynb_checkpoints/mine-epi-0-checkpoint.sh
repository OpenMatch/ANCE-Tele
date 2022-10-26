export DATA_DIR=/data/private/sunsi/dataset/msmarco/rocketqa
export OUTPUT_DIR=/data/private/sunsi/experiments/cocondenser/results
## *************************************
## INPUT
export train_job_name=co-condenser-marco
export infer_job_name=inference.${train_job_name}
## OUTPUT
export new_qry_hn_file_name=qry-neg.${train_job_name}
export new_pos_hn_file_name=pos-neg.${train_job_name}
export new_combine_file_name=qry-pos-neg.${train_job_name}
## *************************************
## *************************************
TOKENIZER=bert-base-uncased
TOKENIZER_ID=bert
SplitNum=10
## *************************************
## ENCODE GPU
ENCODE_CUDA="0,1"
ENCODE_CUDAs=(${ENCODE_CUDA//,/ })
ENCODE_CUDA_NUM=${#ENCODE_CUDAs[@]}
## SEARCH GPU
TOT_CUDA="0,1"
CUDAs=(${TOT_CUDA//,/ })
CUDA_NUM=${#CUDAs[@]}

# **********************************************
# Infer
# **********************************************

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

        CUDA_VISIBLE_DEVICES=${CUDA} python ../encode.py \
        --output_dir ${OUTPUT_DIR}/${infer_job_name} \
        --model_name_or_path ${OUTPUT_DIR}/${train_job_name} \
        --fp16 \
        --per_device_eval_batch_size 1024 \
        --dataloader_num_workers 2 \
        --encode_in_path ${DATA_DIR}/${TOKENIZER_ID}/corpus/split${i}.json \
        --encoded_save_path ${OUTPUT_DIR}/${infer_job_name}/corpus/split${i}.pt &> \
        ${OUTPUT_DIR}/${infer_job_name}/corpus/split${i}.log &&
        ## *************************************
        sleep 3 &
        [ $CUDA_INDEX -eq `expr $ENCODE_CUDA_NUM - 1` ] && wait
    done
done

## *************************************
## Encoding Train query
## *************************************
CUDA_VISIBLE_DEVICES=${CUDAs[-1]} python ../encode.py \
--output_dir ${OUTPUT_DIR}/${infer_job_name} \
--model_name_or_path ${OUTPUT_DIR}/${train_job_name} \
--fp16 \
--q_max_len 32 \
--encode_is_qry \
--per_device_eval_batch_size 2048 \
--dataloader_num_workers 2 \
--encode_in_path ${DATA_DIR}/${TOKENIZER_ID}/query/train.query.json \
--encoded_save_path ${OUTPUT_DIR}/${infer_job_name}/query/train.pt \

## *************************************
## Search Train
## *************************************
CUDA_VISIBLE_DEVICES=${TOT_CUDA} python ../tevatron/faiss_retriever/do_retrieval.py \
--query_reps ${OUTPUT_DIR}/${infer_job_name}/query/train.pt \
--passage_reps ${OUTPUT_DIR}/${infer_job_name}/corpus/'*.pt' \
--index_num ${SplitNum} \
--use_gpu \
--batch_size 1024 \
--save_text \
--depth 200 \
--save_ranking_to ${OUTPUT_DIR}/${infer_job_name}/train.rank.tsv \

## *************************************
## Mine Train Negative
## *************************************
python ../preprocess/build_train_hn.py \
--tokenizer_name ${TOKENIZER} \
--hn_file ${OUTPUT_DIR}/${infer_job_name}/train.rank.tsv \
--qrels ${DATA_DIR}/qrels.train.tsv \
--queries ${DATA_DIR}/train.query.txt \
--collection ${DATA_DIR}/corpus.tsv \
--save_to ${DATA_DIR}/${TOKENIZER_ID}/${new_qry_hn_file_name} \
--depth 200 \
--n_sample 30 \

## *************************************
## Encoding Train-Positives
## *************************************
CUDA_VISIBLE_DEVICES=${CUDAs[-1]} python ../encode.py \
--output_dir ${OUTPUT_DIR}/${infer_job_name} \
--model_name_or_path ${OUTPUT_DIR}/${train_job_name} \
--fp16 \
--per_device_eval_batch_size 1024 \
--dataloader_num_workers 2 \
--encode_in_path ${DATA_DIR}/${TOKENIZER_ID}/query/train.positives.json \
--encoded_save_path ${OUTPUT_DIR}/${infer_job_name}/query/train.positives.pt \

## *************************************
## Search Train-Positives (GPU)
## *************************************
CUDA_VISIBLE_DEVICES=${TOT_CUDA} python ../tevatron/faiss_retriever/do_retrieval.py \
--query_reps ${OUTPUT_DIR}/${infer_job_name}/query/train.positives.pt \
--passage_reps ${OUTPUT_DIR}/${infer_job_name}/corpus/'*.pt' \
--index_num ${SplitNum} \
--use_gpu \
--batch_size 1024 \
--save_text \
--depth 200 \
--save_ranking_to ${OUTPUT_DIR}/${infer_job_name}/train.positives.rank.tsv \


## *************************************
## Mine Train-Positive Negative
## *************************************
python ../preprocess/build_train_hn.py \
--tokenizer_name ${TOKENIZER} \
--hn_file ${OUTPUT_DIR}/${infer_job_name}/train.positives.rank.tsv \
--qrels ${DATA_DIR}/qrels.train.tsv \
--queries ${DATA_DIR}/train.query.txt \
--collection ${DATA_DIR}/corpus.tsv \
--save_to ${DATA_DIR}/${TOKENIZER_ID}/${new_pos_hn_file_name} \
--depth 200 \
--n_sample 30 \

# # *************************************
# # Combine Previous
# # *************************************
python ../preprocess/combine_negative/combine_negative.py \
--data_dir ${DATA_DIR}/${TOKENIZER_ID} \
--input_folder_1 ${new_pos_hn_file_name} \
--input_folder_2 ${new_qry_hn_file_name} \
--output_folder ${new_combine_file_name} \