import os
import json
import argparse
from tqdm import tqdm
from pyserini.search.lucene import LuceneSearcher
from pyserini.eval.evaluate_dpr_retrieval import has_answers, SimpleTokenizer


def get_questions_answers(file_path):
    data_dict = {}
    with open(file_path, "r", encoding="utf-8") as fi:
        for line in tqdm(fi):
            data = json.loads(line)

            data_dict[int(data["qid-num"])] = {
                "question": data["question"], 
                "answers": data["answers"]
            }
    return data_dict
    
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert an TREC run to DPR retrieval result json.')
    parser.add_argument('--topics-file', help='path to a topics file')
    parser.add_argument('--index', required=True, help='Anserini Index that contains raw')
    parser.add_argument('--input', required=True, help='Input TREC run file.')
    parser.add_argument('--store-raw', action='store_true', help='Store raw text of passage')
    parser.add_argument('--regex', action='store_true', default=False, help="regex match")
    parser.add_argument('--output', required=True, help='Output DPR Retrieval json file.')
    parser.add_argument('--depth', type=int, required=True, help='match depth.')
    args = parser.parse_args()

    ## get question & answers 
    qas = get_questions_answers(args.topics_file)

    if os.path.exists(args.index):
        searcher = LuceneSearcher(args.index)
    else:
        searcher = LuceneSearcher.from_prebuilt_index(args.index)
    if not searcher:
        exit()

    retrieval = {}
    tokenizer = SimpleTokenizer()
    with open(args.input) as f_in:
        for line in tqdm(f_in.readlines()):
            question_id, _, doc_id, rank, score, _ = line.strip().split()
            
            if int(rank) > args.depth:
                continue
            
            question_id = int(question_id)
            question = qas[question_id]['question']
            answers = qas[question_id]['answers']
            # if answers[0] == '"':
            #     answers = answers[1:-1].replace('""', '"')
            # answers = eval(answers)
            ctx = json.loads(searcher.doc(doc_id).raw())['contents']
            if question_id not in retrieval:
                retrieval[question_id] = {'question': question, 'answers': answers, 'contexts': []}
            title, text = ctx.split('\n')
            answer_exist = has_answers(text, answers, tokenizer, args.regex)
            if args.store_raw:
                retrieval[question_id]['contexts'].append(
                    {'docid': doc_id,
                     'score': score,
                     'text': ctx,
                     'has_answer': answer_exist}
                )
            else:
                retrieval[question_id]['contexts'].append(
                    {'docid': doc_id, 'score': score, 'has_answer': answer_exist}
                )

    json.dump(retrieval, open(args.output, 'w'), indent=4)