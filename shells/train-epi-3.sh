export DATA_DIR=/data/private/sunsi/dataset/msmarco/rocketqa
export OUTPUT_DIR=/data/private/sunsi/experiments/cocondenser/results
## *************************************
## INPUT
export prev_train_job_name=co-condenser-marco
export train_data=epi-2-20k.combin-prev.qry-pos-neg.co-condenser-marco
## OUTPUT
export train_job_name=epi-3.ance-tele.co-condenser-marco
export infer_job_name=inference.${train_job_name}
## *************************************
## TRAIN GPU
TOT_CUDA="0"
CUDAs=(${TOT_CUDA//,/ })
CUDA_NUM=${#CUDAs[@]}
## ENCODE GPU
ENCODE_CUDA="0,1"
ENCODE_CUDAs=(${ENCODE_CUDA//,/ })
ENCODE_CUDA_NUM=${#ENCODE_CUDAs[@]}
## *************************************
TOKENIZER=bert-base-uncased
TOKENIZER_ID=bert
SplitNum=10
## *************************************

## **********************************************
## Train
## **********************************************
CUDA_VISIBLE_DEVICES=${TOT_CUDA} python ../ancetele/train.py \
--output_dir ${OUTPUT_DIR}/${train_job_name} \
--model_name_or_path ${OUTPUT_DIR}/${prev_train_job_name} \
--fp16 \
--save_steps 20000 \
--train_dir ${DATA_DIR}/${TOKENIZER_ID}/${train_data} \
--per_device_train_batch_size 8 \
--train_n_passages 32 \
--learning_rate 5e-6 \
--num_train_epochs 3 \
--dataloader_num_workers 2 \

# # **********************************************
# # Dist Train
# # **********************************************
# CUDA_VISIBLE_DEVICES=0,1 OMP_NUM_THREADS=2 python -m torch.distributed.launch --nproc_per_node=${CUDA_NUM} --master_port=${PORT} ../ancetele/train.py \
# --output_dir ${OUTPUT_DIR}/${train_job_name} \
# --model_name_or_path ${OUTPUT_DIR}/${prev_train_job_name} \
# --fp16 \
# --save_steps 20000 \
# --train_dir ${DATA_DIR}/${TOKENIZER_ID}/${train_data} \
# --per_device_train_batch_size 4 \
# --train_n_passages 32 \
# --learning_rate 5e-6 \
# --num_train_epochs 3 \
# --dataloader_num_workers 2 \
# --negatives_x_device \


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
CUDA_VISIBLE_DEVICES=${CUDAs[-1]} python ../ancetele/encode.py \
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
CUDA_VISIBLE_DEVICES=${TOT_CUDA} python ../tevatron/faiss_retriever/do_retrieval.py \
--query_reps ${OUTPUT_DIR}/${infer_job_name}/query/qry.pt \
--passage_reps ${OUTPUT_DIR}/${infer_job_name}/corpus/'*.pt' \
--index_num ${SplitNum} \
--use_gpu \
--batch_size 1024 \
--save_text \
--depth 10 \
--save_ranking_to ${OUTPUT_DIR}/${infer_job_name}/dev.rank.tsv \

## *************************************
## Compute Dev MRR@10
## *************************************
python ../driver/score_to_marco.py ${OUTPUT_DIR}/${infer_job_name}/dev.rank.tsv
python ../driver/ms_marco_eval.py ${DATA_DIR}/qrels.dev.small.tsv ${OUTPUT_DIR}/${infer_job_name}/dev.rank.tsv.marco &> \
${OUTPUT_DIR}/${infer_job_name}/dev_mrr.log