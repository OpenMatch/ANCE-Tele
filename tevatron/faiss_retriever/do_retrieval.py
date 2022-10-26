import pickle
import torch
import gc
import time
import numpy as np
import glob
from argparse import ArgumentParser
from itertools import chain
from tqdm import tqdm

from retriever import BaseFaissIPRetriever

import logging
logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)


def search_queries(retriever, q_reps, p_lookup, args):
    if args.batch_size > 0:
        all_scores, all_indices = retriever.batch_search(q_reps, args.depth, args.batch_size)
    else:
        all_scores, all_indices = retriever.search(q_reps, args.depth)

    psg_indices = [[str(p_lookup[x]) for x in q_dd] for q_dd in tqdm(all_indices)]
    psg_indices = np.array(psg_indices)
    return all_scores, psg_indices


def write_ranking(corpus_indices, corpus_scores, q_lookup, ranking_save_file):
    with open(ranking_save_file, 'w') as f:
        for qid, q_doc_scores, q_doc_indices in zip(q_lookup, corpus_scores, corpus_indices):
            score_list = [(s, idx) for s, idx in zip(q_doc_scores, q_doc_indices)]
            score_list = sorted(score_list, key=lambda x: x[0], reverse=True)
            for s, idx in score_list:
                f.write(f'{qid}\t{idx}\t{s}\n')
                

def write_trec(q_lookup, qid2docids, ranking_save_file, depth):
    with open(ranking_save_file, 'w') as f:
        for qid in q_lookup:
            score_list = qid2docids[qid]
            score_list = sorted(score_list, key=lambda x: x[0], reverse=True)
            for s, idx in score_list[:depth]:
                f.write(f'{qid}\t{idx}\t{s}\n')

def pickle_load(path):
    with open(path, 'rb') as f:
        obj = pickle.load(f)
    return obj


def pickle_save(obj, path):
    with open(path, 'wb') as f:
        pickle.dump(obj, f)


def main():
    parser = ArgumentParser()
    parser.add_argument('--query_reps', required=True)
    parser.add_argument('--passage_reps', required=True)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--index_num', type=int, required=True)
    parser.add_argument('--use_gpu', action='store_true')
    parser.add_argument('--depth', type=int, default=1000)
    parser.add_argument('--save_ranking_to', required=True)
    parser.add_argument('--save_text', action='store_true')
    parser.add_argument('--sub_split_num', type=int, default=None)

    args = parser.parse_args()
    
    ## *******************************************
    ## Single Search
    ## *******************************************
    if args.sub_split_num is None:
        index_files = glob.glob(args.passage_reps)
        logger.info(f'Pattern match found {len(index_files)} files; loading them into index.')

        p_reps_0, p_lookup_0 = pickle_load(index_files[0])
        retriever = BaseFaissIPRetriever(p_reps_0, args.use_gpu)

        shards = chain([(p_reps_0, p_lookup_0)], map(pickle_load, index_files[1:]))
        if len(index_files) > 1:
            shards = tqdm(shards, desc='Loading shards into index', total=len(index_files))

        assert len(index_files) == args.index_num

        p_reps = []
        look_up = []
        for _p_reps, p_lookup in shards:
            p_reps.append(_p_reps)
            look_up += p_lookup

        p_reps = np.concatenate(p_reps, axis=0)
        retriever.add(p_reps)

        q_reps, q_lookup = pickle_load(args.query_reps)
        q_reps = q_reps

        logger.info('Index Search Start')
        all_scores, psg_indices = search_queries(retriever, q_reps, look_up, args)
        logger.info('Index Search Finished')

        if args.save_text:
            write_ranking(psg_indices, all_scores, q_lookup, args.save_ranking_to)
        else:
            pickle_save((all_scores, psg_indices), args.save_ranking_to)

    ## *******************************************
    ## Split Search
    ## *******************************************
    else:
        print("split corpus search!")
        
        ## Load qry
        q_reps, q_lookup = pickle_load(args.query_reps)
        q_reps = q_reps
        
        ## Load corpus
        filenames = [[filename, int(filename.split("/")[-1].strip(".pt")[-2:])] for filename in glob.glob(args.passage_reps)]
        sorted_filenames = sorted(filenames, key=lambda item:item[1])
        tot_index_files = [item[0] for item in sorted_filenames]
        
        assert len(tot_index_files) == args.index_num
        logger.info(f'Pattern match found {len(tot_index_files)} files; loading them into index.')
        
        ## container
        merge_qid2docids = {qid:[] for qid in q_lookup}
        
        search_time = round(len(tot_index_files) / args.sub_split_num)
        for search_idx in range(search_time):
            index_files = tot_index_files[args.sub_split_num*search_idx:args.sub_split_num*(search_idx+1)]
            print("searching ", search_idx+1, " total: ", search_time)
            
            p_reps_0, p_lookup_0 = pickle_load(index_files[0])
            retriever = BaseFaissIPRetriever(p_reps_0, args.use_gpu)

            shards = chain([(p_reps_0, p_lookup_0)], map(pickle_load, index_files[1:]))
            if len(index_files) > 1:
                shards = tqdm(shards, desc='Loading shards into index', total=len(index_files))

            assert len(index_files) == len(index_files)

            p_reps = []
            look_up = []
            for _p_reps, p_lookup in shards:
                p_reps.append(_p_reps)
                look_up += p_lookup

            p_reps = np.concatenate(p_reps, axis=0)
            retriever.add(p_reps)

            logger.info('Index Search Start: {}/{}'.format(search_idx+1, search_time))
            sub_scores, sub_psg_indices = search_queries(retriever, q_reps, look_up, args)
            logger.info('Index Search Finished: {}/{}'.format(search_idx+1, search_time))
            
            ## Merge
            for qid, q_doc_scores, q_doc_indices in zip(q_lookup, sub_scores, sub_psg_indices):
                score_list = [(s, idx) for s, idx in zip(q_doc_scores, q_doc_indices)]
                merge_qid2docids[qid].extend(score_list)
                
#                 merge_qid2docids[qid] = sorted(merge_qid2docids[qid], key=lambda x: x[0], reverse=True)
#                 merge_qid2docids[qid] = merge_qid2docids[qid][:args.depth]
                
            del retriever
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            time.sleep(5) # just in case the gpu has not cleaned up the memory
            torch.cuda.reset_peak_memory_stats()


        write_trec(q_lookup, merge_qid2docids, args.save_ranking_to, depth=args.depth)



if __name__ == '__main__':
    main()
