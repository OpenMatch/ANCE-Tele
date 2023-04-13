DATA_DIR=/path/to/dataset
OUTPUT_DIR=/path/to/output
CODE_DIR=/path/to/ANCE-Tele
## *************************************
## Custion Input
export PreTrain=co-condenser ## bert-base, bert-large, condenser, co-condenser, t5-base, t5-v1-1-base
export RepResent=dpr ## dpr, ance, ance-tele, colbert, distil
export FewShotTune=fullshot ## zeroshot, fullshot
export AugmentData=augment-none ## augment-none, augment-msmarco, augment-qg, augment-msmarco-qg
train_job_name=${PreTrain}.${RepResent}.${FewShotTune}.${AugmentData}
## *************************************
export pretrain_model_name=co-condenser-wiki
export train_data=final-split-train ## base + novel
## *************************************
## Train Setup
export TOT_CUDA="1"
CUDAs=(${TOT_CUDA//,/ })
CUDA_NUM=${#CUDAs[@]}
# PORT="1234"
q_max_len=32
p_max_len=256
export TOKENIZER=bert-base-uncased ## Attention: when use bert-large, t5, etc...
TOKENIZER_ID=(${TOKENIZER//-/ })
## *************************************

## **********************************************
## Train
## **********************************************
# CUDA_VISIBLE_DEVICES=${TOT_CUDA} python 
# CUDA_VISIBLE_DEVICES=${TOT_CUDA} OMP_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node=${CUDA_NUM} --master_port=${PORT} 

CUDA_VISIBLE_DEVICES=${TOT_CUDA} python ${CODE_DIR}/ancetele/train.py \
--output_dir ${OUTPUT_DIR}/${train_job_name} \
--model_name_or_path ${OUTPUT_DIR}/${pretrain_model_name} \
--train_dir ${DATA_DIR}/${TOKENIZER_ID}/${train_data} \
--save_steps 500000000 \
--fp16 \
--per_device_train_batch_size 32 \
--train_n_passages 2 \
--learning_rate 5e-6 \
--q_max_len ${q_max_len} \
--p_max_len ${p_max_len} \
--num_train_epochs 40 \
--dataloader_num_workers 1 \
--tensorboard \

# --use_t5_decoder
# --param_efficient bitfit

# # # ********************************************************************
# # # If Your CUDA Memory is not enough, Please set the following augments
# # # ********************************************************************
# --negatives_x_device \
# --grad_cache \
# --gc_q_chunk_size 16 \
# --gc_p_chunk_size 32 \

## Split a batch of queries to several gc_q_chunk_size
## Split a batch of passages to several gc_p_chunk_size