export DATA_DIR=/home/sunsi/dataset/msmarco
export OUTPUT_DIR=/home/sunsi/experiments/msmarco-results
## *************************************
## INPUT
export prev_train_job_name=co-condenser-marco
export train_data=epi-2-tele-neg.msmarco ## Mined Epi-2 Tele-Neg
## OUTPUT
export train_job_name=epi-2.ance-tele.msmarco.checkp-20000
## *************************************
## TRAIN GPUs
TOT_CUDA="0" ## multi-gpus: TOT_CUDA="0,1"
CUDAs=(${TOT_CUDA//,/ })
CUDA_NUM=${#CUDAs[@]}
PORT="1234" ## check the port does not occupied
## *************************************
TOKENIZER=bert-base-uncased
TOKENIZER_ID=bert
SplitNum=10
## *************************************

## **********************************************
## Train (Early Stop)
## **********************************************
CUDA_VISIBLE_DEVICES=${TOT_CUDA} python ../ancetele/train.py \
--output_dir ${OUTPUT_DIR}/${train_job_name} \
--model_name_or_path ${OUTPUT_DIR}/${prev_train_job_name} \
--fp16 \
--save_strategy no \
--early_stop_step 20000 \
--train_dir ${DATA_DIR}/${TOKENIZER_ID}/${train_data} \
--per_device_train_batch_size 8 \
--train_n_passages 32 \
--learning_rate 5e-6 \
--num_train_epochs 3 \
--dataloader_num_workers 2 \


# # **********************************************
# # Dist Train (Early Stop)
# # **********************************************
# CUDA_VISIBLE_DEVICES=${TOT_CUDA} OMP_NUM_THREADS=2 python -m torch.distributed.launch --nproc_per_node=${CUDA_NUM} --master_port=${PORT} ../ancetele/train.py \
# --output_dir ${OUTPUT_DIR}/${train_job_name} \
# --model_name_or_path ${OUTPUT_DIR}/${prev_train_job_name} \
# --fp16 \
# --save_strategy no \
# --early_stop_step 20000 \
# --train_dir ${DATA_DIR}/${TOKENIZER_ID}/${train_data} \
# --per_device_train_batch_size 4 \
# --train_n_passages 32 \
# --learning_rate 5e-6 \
# --num_train_epochs 3 \
# --dataloader_num_workers 2 \
# --negatives_x_device \

# # --train_n_passages 16 or 32 is ok, 16 is faster.


# # # ********************************************************************
# # # If Your CUDA Memory is not enough, Please set the following augments
# # # ********************************************************************
# --grad_cache \
# --gc_q_chunk_size 4 \
# --gc_p_chunk_size 8 \

## Split a batch of queries to several gc_q_chunk_size
## Split a batch of passages to several gc_p_chunk_size