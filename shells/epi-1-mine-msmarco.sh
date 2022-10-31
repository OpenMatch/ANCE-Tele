export DATA_DIR=/home/sunsi/dataset/msmarco/msmarco
export OUTPUT_DIR=/home/sunsi/experiments/msmarco-results
## *************************************
## INPUT/OUTPUT
export train_job_name=co-condenser-marco
export infer_job_name=inference.${train_job_name}
## OUTPUT
export new_ann_hn_file_name=ann-neg.${train_job_name}
export new_la_hn_file_name=la-neg.${train_job_name}
export new_tele_file_name_wo_mom=epi-1-tele-neg.msmarco
## *************************************
## *************************************
TOKENIZER=bert-base-uncased
TOKENIZER_ID=bert
SplitNum=10
## *************************************
## ENCODE Corpus GPUs
ENCODE_CUDA="0,1,2,3,4"
ENCODE_CUDAs=(${ENCODE_CUDA//,/ })
ENCODE_CUDA_NUM=${#ENCODE_CUDAs[@]}
## Search Top-k GPUs
SEARCH_CUDA="0,1,2,3,4"

# ## **********************************************
# ## Infer
# ## **********************************************
# ## Create Folder
# mkdir -p ${OUTPUT_DIR}/${infer_job_name}/corpus
# mkdir -p ${OUTPUT_DIR}/${infer_job_name}/query

# ## Encoding Corpus
# for((tmp=0; tmp<$SplitNum; tmp+=$ENCODE_CUDA_NUM))
# do
#     ## *************************************
#     for((CUDA_INDEX=0; CUDA_INDEX<$ENCODE_CUDA_NUM; CUDA_INDEX++))
#     do
#         ## *************************************
#         if [ $[CUDA_INDEX + $tmp] -eq $SplitNum ]
#         then
#           break 2
#         fi

#         ## *************************************
#         printf -v i "%02g" $[CUDA_INDEX + $tmp] &&
#         CUDA=${ENCODE_CUDAs[$CUDA_INDEX]} &&
#         echo ${OUTPUT_DIR}/${train_job_name} &&
#         echo split-${i} on gpu-${CUDA} &&

#         CUDA_VISIBLE_DEVICES=${CUDA} python ../ancetele/encode.py \
#         --output_dir ${OUTPUT_DIR}/${infer_job_name} \
#         --model_name_or_path ${OUTPUT_DIR}/${train_job_name} \
#         --fp16 \
#         --per_device_eval_batch_size 1024 \
#         --dataloader_num_workers 2 \
#         --encode_in_path ${DATA_DIR}/${TOKENIZER_ID}/corpus/split${i}.json \
#         --encoded_save_path ${OUTPUT_DIR}/${infer_job_name}/corpus/split${i}.pt &> \
#         ${OUTPUT_DIR}/${infer_job_name}/corpus/split${i}.log &&
#         ## *************************************
#         sleep 3 &
#         [ $CUDA_INDEX -eq `expr $ENCODE_CUDA_NUM - 1` ] && wait
#     done
# done

# ## *************************************
# ## Encoding Train-Queries
# ## *************************************
# CUDA_VISIBLE_DEVICES=${ENCODE_CUDAs[-1]} python ../ancetele/encode.py \
# --output_dir ${OUTPUT_DIR}/${infer_job_name} \
# --model_name_or_path ${OUTPUT_DIR}/${train_job_name} \
# --fp16 \
# --q_max_len 32 \
# --encode_is_qry \
# --per_device_eval_batch_size 2048 \
# --dataloader_num_workers 2 \
# --encode_in_path ${DATA_DIR}/${TOKENIZER_ID}/query/train.query.json \
# --encoded_save_path ${OUTPUT_DIR}/${infer_job_name}/query/train.pt \

## *************************************
## Encoding Train-Positives
## *************************************
CUDA_VISIBLE_DEVICES=${ENCODE_CUDAs[-1]} python ../ancetele/encode.py \
--output_dir ${OUTPUT_DIR}/${infer_job_name} \
--model_name_or_path ${OUTPUT_DIR}/${train_job_name} \
--fp16 \
--per_device_eval_batch_size 1024 \
--dataloader_num_workers 2 \
--encode_in_path ${DATA_DIR}/${TOKENIZER_ID}/query/train.positives.json \
--encoded_save_path ${OUTPUT_DIR}/${infer_job_name}/query/train.positives.pt \


## *************************************
## Search Train (GPU)
## *************************************
CUDA_VISIBLE_DEVICES=${SEARCH_CUDA} python ../ancetele/faiss_retriever/do_retrieval.py \
--query_reps ${OUTPUT_DIR}/${infer_job_name}/query/train.pt \
--passage_reps ${OUTPUT_DIR}/${infer_job_name}/corpus/'*.pt' \
--index_num ${SplitNum} \
--use_gpu \
--batch_size 1024 \
--save_text \
--depth 200 \
--save_ranking_to ${OUTPUT_DIR}/${infer_job_name}/train.rank.tsv \
--sub_split_num 5 \
## sub_split_num: if CUDA memory is not enough, set this augments.

## *************************************
## Search Train-Positives (GPU)
## *************************************
CUDA_VISIBLE_DEVICES=${SEARCH_CUDA} python ../ancetele/faiss_retriever/do_retrieval.py \
--query_reps ${OUTPUT_DIR}/${infer_job_name}/query/train.positives.pt \
--passage_reps ${OUTPUT_DIR}/${infer_job_name}/corpus/'*.pt' \
--index_num ${SplitNum} \
--use_gpu \
--batch_size 1024 \
--save_text \
--depth 200 \
--save_ranking_to ${OUTPUT_DIR}/${infer_job_name}/train.positives.rank.tsv \
--sub_split_num 5 \
## sub_split_num: if CUDA memory is not enough, set this augments.

## *************************************
## Mine Train Negative
## *************************************
python ../preprocess/build_train_hn.py \
--tokenizer_name ${TOKENIZER} \
--hn_file ${OUTPUT_DIR}/${infer_job_name}/train.rank.tsv \
--qrels ${DATA_DIR}/qrels.train.tsv \
--queries ${DATA_DIR}/train.query.txt \
--collection ${DATA_DIR}/corpus.tsv \
--save_to ${DATA_DIR}/${TOKENIZER_ID}/${new_ann_hn_file_name} \
--depth 200 \
--n_sample 30 \

## *************************************
## Mine Train-Positive Negative
## *************************************
python ../preprocess/build_train_hn.py \
--tokenizer_name ${TOKENIZER} \
--hn_file ${OUTPUT_DIR}/${infer_job_name}/train.positives.rank.tsv \
--qrels ${DATA_DIR}/qrels.train.tsv \
--queries ${DATA_DIR}/train.query.txt \
--collection ${DATA_DIR}/corpus.tsv \
--save_to ${DATA_DIR}/${TOKENIZER_ID}/${new_la_hn_file_name} \
--depth 200 \
--n_sample 30 \

# # *************************************
# # Combine ANN + LA Negatives
# # *************************************
python ../preprocess/combine_marco_negative.py \
--data_dir ${DATA_DIR}/${TOKENIZER_ID} \
--input_folder_1 ${new_la_hn_file_name} \
--input_folder_2 ${new_ann_hn_file_name} \
--output_folder ${new_tele_file_name_wo_mom} \