## *************************************************
## Download Data
## *************************************************
DATA_DIR=/data/private/sunsi/dataset/msmarco/rocketqa ## Save Data Folder
cd ${DATA_DIR}

wget --no-check-certificate https://rocketqa.bj.bcebos.com/corpus/marco.tar.gz
tar -zxf marco.tar.gz
rm -rf marco.tar.gz
cd marco

wget https://msmarco.blob.core.windows.net/msmarcoranking/qrels.train.tsv -O qrels.train.tsv
## Please download original dev-qrels into this folder: qrels.dev.small.tsv

join  -t "$(echo -en '\t')"  -e '' -a 1  -o 1.1 2.2 1.2  <(sort -k1,1 para.txt) <(sort -k1,1 para.title.txt) | sort -k1,1 -n > corpus.tsv

cd ../
mv ./marco/* ./
rm -rf marco

## *************************************************
## Download Pretrain Model
## *************************************************
OUTPUT_DIR=/data/private/sunsi/experiments/cocondenser/results ## Save Model Folder
cd ${OUTPUT_DIR}

mkdir co-condenser-marco
cd co-condenser-marco

wget -c https://huggingface.co/Luyu/co-condenser-marco/resolve/main/config.json
wget -c https://huggingface.co/Luyu/co-condenser-marco/resolve/main/pytorch_model.bin
wget -c https://huggingface.co/Luyu/co-condenser-marco/resolve/main/special_tokens_map.json
wget -c https://huggingface.co/Luyu/co-condenser-marco/resolve/main/tokenizer_config.json
wget -c https://huggingface.co/Luyu/co-condenser-marco/resolve/main/vocab.txt