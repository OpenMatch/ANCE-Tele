import os
import json
from tqdm import tqdm
from dataclasses import dataclass
from argparse import ArgumentParser
from transformers import AutoTokenizer
from transformers import PreTrainedTokenizer


@dataclass
class QuestionPreProcessor:
    tokenizer: PreTrainedTokenizer
    separator: str = '\t'

    def process_line(self, line: str):
        xx = json.loads(line.strip("\n"))
        text_id, text = xx["qid"], xx["question"]
        text_encoded = self.tokenizer.encode(
            self.tokenizer.sep_token.join([text]),
            add_special_tokens=False,
            truncation=True
        )
        encoded = {
            'text_id': text_id,
            'text': text_encoded
        }
        return json.dumps(encoded)


parser = ArgumentParser()
parser.add_argument('--tokenizer_name', required=True)
parser.add_argument('--query_file', required=True)
parser.add_argument('--save_to', required=True)
args = parser.parse_args()

tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, use_fast=True)
processor = QuestionPreProcessor(tokenizer=tokenizer)

with open(args.query_file, 'r') as f:
    lines = f.readlines()

os.makedirs(os.path.split(args.save_to)[0], exist_ok=True)
with open(args.save_to, 'w') as jfile:
    for x in tqdm(lines):
        q = processor.process_line(x)
        jfile.write(q + '\n')
