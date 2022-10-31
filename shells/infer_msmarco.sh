export DATA_DIR=/home/sunsi/dataset/msmarco/msmarco
export OUTPUT_DIR=/home/sunsi/experiments/msmarco-results
## *************************************
## INPUT
export train_job_name=ance-tele_msmarco_qry-psg-encoder
export infer_job_name=inference.${train_job_name}
## *************************************
## ENCODE Psg GPUs
ENCODE_CUDA="0,1,2,3,4" ## ENCODE_CUDA="0"
ENCODE_CUDAs=(${ENCODE_CUDA//,/ })
ENCODE_CUDA_NUM=${#ENCODE_CUDAs[@]}
## Search Top-k GPUs
SEARCH_CUDA="0,1,2,3,4"
## *************************************
TOKENIZER=bert-base-uncased
TOKENIZER_ID=bert
SplitNum=10
## *************************************
## Create Folder
mkdir -p ${OUTPUT_DIR}/${infer_job_name}/corpus
mkdir -p ${OUTPUT_DIR}/${infer_job_name}/query

## **********************************************
## Infer
## **********************************************
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
## Encoding Dev query
## *************************************
CUDA_VISIBLE_DEVICES=${ENCODE_CUDAs[-1]} python ../ancetele/encode.py \
--output_dir ${OUTPUT_DIR}/${infer_job_name} \
--model_name_or_path ${OUTPUT_DIR}/${train_job_name} \
--fp16 \
--q_max_len 32 \
--encode_is_qry \
--per_device_eval_batch_size 2048 \
--encode_in_path ${DATA_DIR}/${TOKENIZER_ID}/query/dev.query.json \
--encoded_save_path ${OUTPUT_DIR}/${infer_job_name}/query/qry.pt \

## *************************************
## Search Dev (GPU/CPU)
## *************************************
CUDA_VISIBLE_DEVICES=${SEARCH_CUDA} python ../ancetele/faiss_retriever/do_retrieval.py \
--query_reps ${OUTPUT_DIR}/${infer_job_name}/query/qry.pt \
--passage_reps ${OUTPUT_DIR}/${infer_job_name}/corpus/'*.pt' \
--index_num ${SplitNum} \
--use_gpu \
--batch_size 1024 \
--save_text \
--depth 10 \
--save_ranking_to ${OUTPUT_DIR}/${infer_job_name}/dev.rank.tsv \
# --sub_split_num 5 \


## *************************************
## Compute Dev MRR@10
## *************************************
python ../driver/score_to_marco.py ${OUTPUT_DIR}/${infer_job_name}/dev.rank.tsv
python ../driver/ms_marco_eval.py ${DATA_DIR}/qrels.dev.small.tsv ${OUTPUT_DIR}/${infer_job_name}/dev.rank.tsv.marco &> \
${OUTPUT_DIR}/${infer_job_name}/dev_mrr.log