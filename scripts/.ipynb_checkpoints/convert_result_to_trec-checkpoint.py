from tqdm import tqdm
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('--input')
args = parser.parse_args()

output_file = args.input + ".teIn"
with open(args.input) as f_in, open(output_file, 'w') as f_out:
    cur_qid = None
    rank = 0
    for line in tqdm(f_in):
        qid, docid, score = line.split()
        if cur_qid != qid:
            cur_qid = qid
            rank = 0
        rank += 1
        f_out.write(f'{qid} Q0 {docid} {rank} {score} dense\n')