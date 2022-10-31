import os
import csv
import json
import random
import datasets
from tqdm import tqdm
from datetime import datetime
from multiprocessing import Pool
from argparse import ArgumentParser
from transformers import AutoTokenizer

def read_qrel(relevance_file):
    qrel = {}
    with open(relevance_file, encoding='utf8') as f:
        tsvreader = csv.reader(f, delimiter="\t")
        for [topicid, _, docid, rel] in tsvreader:
            assert rel == "1"
            if topicid in qrel:
                qrel[topicid].append(docid)
            else:
                qrel[topicid] = [docid]
    return qrel
    
def get_passage(p, collection, tokenizer, max_length=128):
    entry = collection[int(p)]
    title = entry['title']
    title = "" if title is None else title
    body = entry['text']
    content = title + tokenizer.sep_token + body

    passage_encoded = tokenizer.encode(
        content,
        add_special_tokens=False,
        max_length=max_length,
        truncation=True
    )
    return passage_encoded
    
if __name__ == "__main__":
    
    parser = ArgumentParser()
    parser.add_argument('--data_dir', required=True)
    parser.add_argument('--tokenizer_name', required=True)
    parser.add_argument('--save_to', required=True)
    parser.add_argument('--truncate', type=int, default=128)
    args = parser.parse_args()
    
    queries_path = os.path.join(args.data_dir, "train.query.txt")
    collection_path = os.path.join(args.data_dir, "corpus.tsv")
    
    qrel = read_qrel(os.path.join(args.data_dir, "qrels.train.tsv"))
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, use_fast=True)
    
    collection = datasets.load_dataset(
        'csv',
        data_files=collection_path,
        cache_dir=collection_path+".cache",
        column_names=['text_id', 'title', 'text'],
        delimiter='\t',
    )['train']
    
    with open(args.save_to, 'w') as jfile:
        for qid, docids in tqdm(qrel.items()):
            text_encoded = get_passage(docids[0], collection, tokenizer, max_length=args.truncate)
            encoded = {
                'text_id': qid,
                'text': text_encoded
            }
            jfile.write(json.dumps(encoded) + '\n')