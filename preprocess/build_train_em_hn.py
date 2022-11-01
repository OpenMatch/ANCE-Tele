import os
import json
import random
import datasets
from tqdm import tqdm
from datetime import datetime
from multiprocessing import Pool
from dataclasses import dataclass
from argparse import ArgumentParser
from transformers import AutoTokenizer
from transformers import PreTrainedTokenizer
from pyserini.eval.evaluate_dpr_retrieval import SimpleTokenizer, has_answers


@dataclass
class WikiTrainPreProcessor:
    query_file: str
    collection_file: str
    tokenizer: PreTrainedTokenizer
    
    max_length: int = 256
    columns = ['text_id', 'text', 'title'] ## psgs_w100.tsv split
    title_field = 'title'
    text_field = 'text'

    def __post_init__(self):
        
        ## qid: qry, ans, pos
        self.queries = self.read_queries(self.query_file)
        
        ## Corpus
        self.collection = datasets.load_dataset(
            'csv',
            data_files=self.collection_file,
            cache_dir=self.collection_file+".cache",
            column_names=self.columns,
            delimiter='\t',
        )['train']
        
        
    @staticmethod
    def read_queries(query_file):
        queries = datasets.load_dataset(
            'json',
            'default',
            data_files=query_file,
            cache_dir=query_file+".cache",
        )['train']
        
        qid2queries = {}
        for item in queries:
            qid = int(item["qid"])
            qid2queries[qid] = item
        return qid2queries
    
    
    def get_query(self, qid):
        query_encoded = self.tokenizer.encode(
            self.queries[qid]["question"],
            add_special_tokens=False,
            max_length=self.max_length,
            truncation=True
        )
        return query_encoded
    

    def get_passage(self, p):
        entry = self.collection[p]
        title = entry[self.title_field]
        title = "" if title is None else title
        body = entry[self.text_field]
        content = title + self.tokenizer.sep_token + body

        passage_encoded = self.tokenizer.encode(
            content,
            add_special_tokens=False,
            max_length=self.max_length,
            truncation=True
        )

        return passage_encoded
    
    
    def tokenize_passage(self, title, body):
        title = "" if title is None else title
        content = title + self.tokenizer.sep_token + body

        passage_encoded = self.tokenizer.encode(
            content,
            add_special_tokens=False,
            max_length=self.max_length,
            truncation=True
        )
        return passage_encoded

    
    def process_one(self, train):
        q, origin_pp, pp, nn = train

        if len(origin_pp) > 0:
            positives = [
                self.tokenize_passage(
                origin_p["title"], 
                origin_p["text"]) for origin_p in origin_pp
            ]
        else:
            positives = [self.get_passage(p) for p in pp]
        
        train_example = {
            'qid': q,
            'query': self.get_query(q),
            'positives': positives,
            'negatives': [self.get_passage(n) for n in nn],
        }

        return json.dumps(train_example)


    
def load_ranking(
    rank_file, 
    queries, 
    collection, 
    em_tokenizer, 
    n_sample, 
    depth, 
    minimum_negatives=1
):
    with open(rank_file) as rf:
        lines = iter(rf)
        q_0, p_0, _ = next(lines).strip().split()
        q_0 = int(q_0)
        p_0 = int(p_0)
        
        curr_q = q_0
        content = collection[p_0]['text'] ## only match main body text not title
        answers = queries[q_0]["answers"]
        
        negatives = []
        new_positives = []
        if not has_answers(content, answers, em_tokenizer, regex=False):
            negatives.append(p_0)
        else:
            new_positives.append(p_0)

        while True:
            try:
                q, p, _ = next(lines).strip().split()
                q = int(q)
                p = int(p)
                ## *************************
                ## Time to finish curr_q !
                ## *************************
                if q != curr_q:

                    ## Positive
                    origin_positives = queries[curr_q]["positive_ctxs"] ## {"title": , "text", }
                    
                    ## Negative
                    negatives = negatives[:depth]
                    random.shuffle(negatives)
                    
                    if (len(origin_positives) + len(new_positives)) >= 1 and len(negatives) >= minimum_negatives:
                        yield curr_q, origin_positives, new_positives[:1], negatives[:n_sample]
                        
                    ## *************************
                    ## Time to next q !
                    ## *************************
                    curr_q = q
                    content = collection[p]['text']
                    answers = queries[q]["answers"]
                    
                    negatives = []
                    new_positives = []
                    if not has_answers(content, answers, em_tokenizer, regex=False):
                        negatives.append(p)
                    else:
                        new_positives.append(p)
                        
                ## *************************
                ## Continue curr_q ...
                ## *************************
                else:
                    content = collection[p]['text']
                    answers = queries[q]["answers"]
                    if not has_answers(content, answers, em_tokenizer, regex=False):
                        negatives.append(p)
                    else:
                        new_positives.append(p)
                        
            ## *************************
            ## END
            ## *************************
            except StopIteration:

                ## Positive
                origin_positives = queries[curr_q]["positive_ctxs"] ## {"title": , "text", }

                ## Negative
                negatives = negatives[:depth]
                random.shuffle(negatives)
                
                if (len(origin_positives) + len(new_positives)) >= 1 and len(negatives) >= minimum_negatives:
                    yield curr_q, origin_positives, new_positives[:1], negatives[:n_sample]
                return

if __name__ == "__main__":

    random.seed(datetime.now())
    parser = ArgumentParser()
    parser.add_argument('--tokenizer_name', required=True)
    parser.add_argument('--input_file', required=True)
    parser.add_argument('--queries', required=True)
    parser.add_argument('--collection', required=True)
    parser.add_argument('--save_to', required=True)
    parser.add_argument('--mark', type=str, default="hn")

    parser.add_argument('--truncate', type=int, default=156)
    parser.add_argument('--n_sample', type=int, default=30)
    parser.add_argument('--depth', type=int, default=200)
    parser.add_argument('--mp_chunk_size', type=int, default=500)
    parser.add_argument('--shard_size', type=int, default=45000)
    parser.add_argument('--gen_pos_file', type=str, default=None)


    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_name, 
        use_fast=True
    )

    processor = WikiTrainPreProcessor(
        query_file=args.queries,
        collection_file=args.collection,
        tokenizer=tokenizer,
        max_length=args.truncate,
    )

    counter = 0
    shard_id = 0
    f = None
    os.makedirs(args.save_to, exist_ok=True)


    pbar = tqdm(
        load_ranking(
            rank_file=args.input_file, 
            queries=processor.queries,
            collection=processor.collection,
            em_tokenizer=SimpleTokenizer(),
            n_sample=args.n_sample, 
            depth=args.depth,
        )
    )

    with Pool() as p:
        for x in p.imap(processor.process_one, pbar, chunksize=args.mp_chunk_size):
            counter += 1
            if f is None:
                f = open(os.path.join(args.save_to, f'split{shard_id:02d}.{args.mark}.json'), 'w')
                pbar.set_description(f'split - {shard_id:02d}')
            f.write(x + '\n')

            if counter == args.shard_size:
                f.close()
                f = None
                shard_id += 1
                counter = 0

    if f is not None:
        f.close()


    if args.gen_pos_file:
        file_list = [os.path.join(args.save_to, listx) for listx in os.listdir(args.save_to) \
                     if "json" in listx and "cache" not in listx]
        with open(args.gen_pos_file, "w", encoding="utf-8") as fw:
            for file_path in tqdm(file_list):
                with open(file_path, "r", encoding="utf-8") as fi:
                    for line in fi:
                        data = json.loads(line)

                        qid = data["qid"]
                        positives = data["positives"][0]

                        save_item = {"text_id":qid, "text":positives}
                        fw.write(json.dumps(save_item) + '\n')
