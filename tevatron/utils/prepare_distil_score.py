import datasets
from tqdm import tqdm
from numpy import ndarray
from multiprocessing import Pool




def multiprocessing_distil_scores(
    examples: datasets.Dataset,                                       
    tot_scores: ndarray,
    encoded_save_path: str,
    chunksize: int = 500,
    shard_size: int = 45000,
):

    counter = 0
    shard_id = 0
    f = None
    pbar = tqdm(prepare_distil_scores(examples, tot_scores))                        
    with Pool() as p:
        for x in p.imap(
            func=write_distil_scores, 
            iterable=pbar, 
            chunksize=chunksize
        ):
            counter += 1
            if f is None:
                f = open(os.path.join(encoded_save_path, f'split{shard_id:02d}.hn.json'), 'w')
                pbar.set_description(f'split - {shard_id:02d}')
            f.write(x + '\n')

            if counter == shard_size:                                                                
                f.close()
                f = None
                shard_id += 1
                counter = 0

    if f is not None:
        f.close()
    

def prepare_distil_scores(examples, tot_scores):
    
    iter_examples = iter(examples)
    iter_scores = iter(tot_scores)
    while True:
        try:
            ex = next(iter_examples)
            query = ex['query']
            positives = ex['positives']
            negatives = ex['negatives']
            
            pos_scores = []
            for _ in positives:
                pos_scores.append(next(iter_scores))
                
            neg_scores = []
            for _ in negatives:
                neg_scores.append(next(iter_scores))
            
            yield query, positives, negatives, pos_scores, neg_scores
            
        except StopIteration:
            return

        
def write_distil_scores(item):
    qry, positives, negatives, pos_scores, neg_scores = item
    train_example = {
        'query': qry,
        'positives': positives,
        'negatives': negatives,
        'pos_scores': pos_scores,
        'neg_scores': neg_scores
    }

    return json.dumps(train_example)


# def multiprocessing_distil_scores(
#     examples: datasets.Dataset,                                       
#     tot_scores: ndarray,
#     encoded_save_path: str,
#     chunksize: int = 500,
#     shard_size: int = 45000,
# ):

#     counter = 0
#     shard_id = 0
#     f = None
#     pbar = tqdm(prepare_distil_scores(examples, tot_scores))                        
#     with Pool() as p:
#         for x in p.imap(
#             func=write_distil_scores, 
#             iterable=pbar, 
#             chunksize=chunksize
#         ):
#             counter += 1
#             if f is None:
#                 f = open(os.path.join(encoded_save_path, f'split{shard_id:02d}.hn.json'), 'w')
#                 pbar.set_description(f'split - {shard_id:02d}')
#             f.write(x + '\n')

#             if counter == shard_size:                                                                
#                 f.close()
#                 f = None
#                 shard_id += 1
#                 counter = 0

#     if f is not None:
#         f.close()
    

# def prepare_distil_scores(examples, tot_scores):
    
#     iter_examples = iter(examples)
#     iter_scores = iter(tot_scores)
#     while True:
#         try:
#             ex = next(iter_examples)
#             query = ex['query']
#             positives = ex['positives']
#             negatives = ex['negatives']
            
#             pos_scores = []
#             for _ in positives:
#                 pos_scores.append(next(iter_scores))
                
#             neg_scores = []
#             for _ in negatives:
#                 neg_scores.append(next(iter_scores))
            
#             yield query, positives, negatives, pos_scores, neg_scores
            
#         except StopIteration:
#             return

        
# def write_distil_scores(item):
#     qry, positives, negatives, pos_scores, neg_scores = item
#     train_example = {
#         'query': qry,
#         'positives': positives,
#         'negatives': negatives,
#         'pos_scores': pos_scores,
#         'neg_scores': neg_scores
#     }

#     return json.dumps(train_example)
        
