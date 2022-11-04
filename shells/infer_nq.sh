export DATA_DIR=/home/sunsi/dataset/nq
export OUTPUT_DIR=/home/sunsi/experiments/nq-results
export CORPUS_DATA_DIR=/home/sunsi/dataset/wikipedia-corpus-index
export pyserini_eval_topics=dpr-nq-test
## *************************************
## INPUT/OUTPUT
export qry_encoder_name=ance-tele_nq_qry-encoder
export psg_encoder_name=ance-tele_nq_psg-encoder
export infer_job_name=inference.ance-tele.nq
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
        --model_name_or_path ${OUTPUT_DIR}/${psg_encoder_name} \
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
## Encode [Test-Query]
## *************************************
CUDA_VISIBLE_DEVICES=${ENCODE_CUDAs[-1]} python ../ancetele/encode.py \
--output_dir ${OUTPUT_DIR}/${infer_job_name} \
--model_name_or_path ${OUTPUT_DIR}/${qry_encoder_name} \
--fp16 \
--q_max_len ${q_max_len} \
--encode_is_qry \
--per_device_eval_batch_size 1024 \
--encode_in_path ${DATA_DIR}/${TOKENIZER_ID}/query/test.query.json \
--encoded_save_path ${OUTPUT_DIR}/${infer_job_name}/query/qry.pt \

## *************************************
## Search [Test]
## *************************************
CUDA_VISIBLE_DEVICES=${SEARCH_CUDA} python ../ancetele/faiss_retriever/do_retrieval.py \
--query_reps ${OUTPUT_DIR}/${infer_job_name}/query/qry.pt \
--passage_reps ${OUTPUT_DIR}/${infer_job_name}/corpus/'*.pt' \
--index_num ${SplitNum} \
--batch_size 1024 \
--use_gpu \
--save_text \
--depth 100 \
--save_ranking_to ${OUTPUT_DIR}/${infer_job_name}/test.rank.tsv \
# --sub_split_num 5 \
## if CUDA memory is not enough, set this augment.


## *************************************
## Eval [Test]
## *************************************
python ../scripts/convert_result_to_trec.py --input ${OUTPUT_DIR}/${infer_job_name}/test.rank.tsv \

python -m pyserini.eval.convert_trec_run_to_dpr_retrieval_run \
--topics ${pyserini_eval_topics} \
--index ${CORPUS_DATA_DIR}/index-wikipedia-dpr-20210120-d1b9e6 \
--input ${OUTPUT_DIR}/${infer_job_name}/test.rank.tsv.teIn \
--output ${OUTPUT_DIR}/${infer_job_name}/test.rank.tsv.json \

python -m pyserini.eval.evaluate_dpr_retrieval \
--retrieval ${OUTPUT_DIR}/${infer_job_name}/test.rank.tsv.json --topk 5 20 100 &> ${OUTPUT_DIR}/${infer_job_name}/test-hits.log

## The Test R@5/20/100 resuls are saved in test-hits.log