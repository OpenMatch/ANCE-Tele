export DATA_DIR=/home/sunsi/dataset/triviaqa
export OUTPUT_DIR=/home/sunsi/experiments/triviaqa-results
## *************************************
## INPUT
export prev_train_job_name=co-condenser-wiki
export train_data=epi-3-tele-neg.triviaqa  ## Mined Epi-3 Tele-Neg
## OUTPUT
export train_job_name=epi-3.ance-tele.triviaqa
## *************************************
## TRAIN GPUs
TOT_CUDA="0,1,2,3"
CUDAs=(${TOT_CUDA//,/ })
CUDA_NUM=${#CUDAs[@]}
PORT="1234" ## check the port does not occupied
## *************************************
## Length SetUp
export q_max_len=32
export p_max_len=156
## *************************************
TOKENIZER=bert-base-uncased
TOKENIZER_ID=bert
## *************************************

# **********************************************
# Dist Train
# **********************************************
CUDA_VISIBLE_DEVICES=${TOT_CUDA} OMP_NUM_THREADS=2 python -m torch.distributed.launch --nproc_per_node=${CUDA_NUM} --master_port=${PORT} ../ancetele/train.py \
--output_dir ${OUTPUT_DIR}/${train_job_name} \
--model_name_or_path ${OUTPUT_DIR}/${prev_train_job_name} \
--fp16 \
--save_steps 2000 \
--train_dir ${DATA_DIR}/${TOKENIZER_ID}/${train_data} \
--per_device_train_batch_size 32 \
--train_n_passages 12 \
--learning_rate 5e-6 \
--num_train_epochs 40 \
--q_max_len ${q_max_len} \
--p_max_len ${p_max_len} \
--dataloader_num_workers 2 \
--untie_encoder \
--negatives_x_device \
--positive_passage_no_shuffle \


# # # ********************************************************************
# # # If Your CUDA Memory is not enough, Please set the following augments
# # # ********************************************************************
# --grad_cache \
# --gc_q_chunk_size 16 \
# --gc_p_chunk_size 128 \

## Split a batch of queries to several gc_q_chunk_size
## Split a batch of passages to several gc_p_chunk_size