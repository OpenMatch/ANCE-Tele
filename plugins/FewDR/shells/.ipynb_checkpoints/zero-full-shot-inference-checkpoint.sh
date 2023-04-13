DATA_DIR=/path/to/dataset
OUTPUT_DIR=/path/to/output
CODE_DIR=/path/to/ANCE-Tele
## *************************************
## Custion Input
export PreTrain=co-condenser ## bert-base, bert-large, condenser, co-condenser, t5-base, t5-v1-1-base
export RepResent=dpr ## dpr, ance, ance-tele, colbert, distil
export FewShotTune=zeroshot ## zeroshot // fullshot
export AugmentData=augment-none ## augment-none, augment-msmarco, augment-qg, augment-msmarco-qg
train_job_name=${PreTrain}.${RepResent}.${FewShotTune}.${AugmentData}
infer_job_name=inference.${train_job_name}
## *************************************
## GPU Setup
export TOT_CUDA="0,1"
CUDAs=(${TOT_CUDA//,/ })
CUDA_NUM=${#CUDAs[@]}
## FAISS_CUDA can be different from TOT_CUDA
# FAISS_CUDA=$TOT_CUDA
export FAISS_CUDA="0,1"
## *************************************
## Eval Setup
split_data_stg=final-split
q_max_len=32
p_max_len=256
eval_batch_size=2048 ## A100 can be 2048; 2080 can be 512
export TOKENIZER=bert-base-uncased ## Attention: when use bert-large, t5, etc...
TOKENIZER_ID=(${TOKENIZER//-/ })
SplitNum=20
## *************************************


## **************************************************************************
## Starting Inference
mkdir -p ${OUTPUT_DIR}/${infer_job_name}/corpus
mkdir -p ${OUTPUT_DIR}/${infer_job_name}/query

## ************************************
## (1) Encoding Corpus
## ************************************
for((tmp=0; tmp<$SplitNum; tmp+=$CUDA_NUM))
do
    ## *************************************
    for((CUDA_INDEX=0; CUDA_INDEX<$CUDA_NUM; CUDA_INDEX++))
    do
        ## *************************************
        if [ $[CUDA_INDEX + $tmp] -lt $SplitNum ]
        then
            ## *************************************
            printf -v i "%02g" $[CUDA_INDEX + $tmp] &&
            CUDA=${CUDAs[$CUDA_INDEX]} &&
            echo ${OUTPUT_DIR}/${train_job_name} &&
            echo split-${i} on gpu-${CUDA} &&

            export CUDA_VISIBLE_DEVICES=${CUDA} && python ${CODE_DIR}/ancetele/encode.py \
            --output_dir ${OUTPUT_DIR}/${infer_job_name} \
            --model_name_or_path ${OUTPUT_DIR}/${train_job_name} \
            --fp16 \
            --per_device_eval_batch_size ${eval_batch_size} \
            --dataloader_num_workers 1 \
            --p_max_len ${p_max_len} \
            --encode_in_path ${DATA_DIR}/${TOKENIZER_ID}/corpus/split${i}.json \
            --encoded_save_path ${OUTPUT_DIR}/${infer_job_name}/corpus/split${i}.pt &> \
            ${OUTPUT_DIR}/${infer_job_name}/corpus/split${i}.log &
        fi
        ## *************************************
        sleep 3 &
        [ $CUDA_INDEX -eq `expr $CUDA_NUM - 1` ] && wait
    done
done


## *************************************
## (2) Encode Query
## *************************************
CUDA_VISIBLE_DEVICES=${CUDAs[0]} python ${CODE_DIR}/ancetele/encode.py \
--output_dir ${OUTPUT_DIR}/${infer_job_name} \
--model_name_or_path ${OUTPUT_DIR}/${train_job_name} \
--fp16 \
--q_max_len ${q_max_len} \
--encode_is_qry \
--per_device_eval_batch_size ${eval_batch_size} \
--encode_in_path ${DATA_DIR}/${TOKENIZER_ID}/query/test.query.json \
--encoded_save_path ${OUTPUT_DIR}/${infer_job_name}/query/test.query.pt \

## *************************************
## (3) Search
## *************************************
CUDA_VISIBLE_DEVICES=${FAISS_CUDA} python ${CODE_DIR}/ancetele/faiss_retriever/do_retrieval.py \
--query_reps ${OUTPUT_DIR}/${infer_job_name}/query/test.query.pt \
--passage_reps ${OUTPUT_DIR}/${infer_job_name}/corpus/'*.pt' \
--index_num ${SplitNum} \
--batch_size 2048 \
--use_gpu \
--save_text \
--depth 10 \
--save_ranking_to ${OUTPUT_DIR}/${infer_job_name}/test.rank.trec \
--save_format trec \
--sub_split_num 5 \

# ## *************************************
# ## (4) Eval
# ## *************************************
python ${CODE_DIR}/scripts/convert_trec_run_to_match_ans_run.py \
--topics-file ${DATA_DIR}/tot-qid_query_answer_positive.jsonl \
--index ${DATA_DIR}/wikipedia-corpus-index/index-wikipedia-dpr-20210120-d1b9e6 \
--input ${OUTPUT_DIR}/${infer_job_name}/test.rank.trec \
--output ${OUTPUT_DIR}/${infer_job_name}/test.rank.json \
--depth 10 \

python ${CODE_DIR}/scripts/evaluate_fewrel_retrieval.py \
--topic_class ${DATA_DIR}/qid2num.json \
--retrieval ${OUTPUT_DIR}/${infer_job_name}/test.rank.json \
--topk 10 \
--split_dataset ${DATA_DIR}/data-split/${split_data_stg}/split-stg.json \


# ## *************************************
# ## (5) delete embedding
# ## *************************************
if [ -s ${OUTPUT_DIR}/${infer_job_name}/test.rank.trec ]
then
    echo "Successfully saved trec file! Delete embedding files :) "
    rm -rf ${OUTPUT_DIR}/${infer_job_name}/corpus
    rm -rf ${OUTPUT_DIR}/${infer_job_name}/query
else
    echo "There are some troubles in saving trec file ..."
fi



