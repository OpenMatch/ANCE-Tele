import os
import json
from tqdm import tqdm
from multiprocessing import Pool
from dataclasses import dataclass
from argparse import ArgumentParser
from transformers import AutoTokenizer, PreTrainedTokenizer

@dataclass
class WikiCollectionPreProcessor:
    tokenizer: PreTrainedTokenizer
    separator: str = '\t'
    max_length: int = 256

    def process_line(self, line: str):
        xx = line.strip().split(self.separator)
        text_id, body, title = xx[0], xx[1], xx[2]
        
        if text_id == "id":
            return None
        
        title = "" if title is None else title
        
        text = title + self.tokenizer.sep_token + body
        text_encoded = self.tokenizer.encode(
            text,
            add_special_tokens=False,
            max_length=self.max_length,
            truncation=True
        )
        
        encoded = {
            'text_id': text_id,
            'text': text_encoded
        }
        return json.dumps(encoded)

parser = ArgumentParser()
parser.add_argument('--tokenizer_name', required=True)
parser.add_argument('--truncate', type=int, default=256)
parser.add_argument('--file', required=True)
parser.add_argument('--save_to', required=True)
parser.add_argument('--n_splits', type=int, default=20)

args = parser.parse_args()

tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, use_fast=True)
processor = WikiCollectionPreProcessor(tokenizer=tokenizer, max_length=args.truncate)

with open(args.file, 'r') as f:
    lines = f.readlines()

n_lines = len(lines)
if n_lines % args.n_splits == 0:
    split_size = int(n_lines / args.n_splits)
else:
    split_size = int(n_lines / args.n_splits) + 1


os.makedirs(args.save_to, exist_ok=True)
with Pool() as p:
    for i in range(args.n_splits):
        with open(os.path.join(args.save_to, f'split{i:02d}.json'), 'w') as f:
            pbar = tqdm(lines[i*split_size: (i+1)*split_size])
            pbar.set_description(f'split - {i:02d}')
            for jitem in p.imap(processor.process_line, pbar, chunksize=500):
                if jitem is not None:
                    f.write(jitem + '\n')


