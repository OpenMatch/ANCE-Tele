import os
import json
import numpy as np
from tqdm import tqdm
from argparse import ArgumentParser

if __name__ == "__main__":
    
    parser = ArgumentParser()
    parser.add_argument('--data_dir', required=True)
    parser.add_argument('--input_folder_1', required=True)
    parser.add_argument('--input_folder_2', required=True)
    parser.add_argument('--output_folder', required=True)
    args = parser.parse_args()
    
    input_path_1 = os.path.join(args.data_dir, args.input_folder_1)
    input_path_2 = os.path.join(args.data_dir, args.input_folder_2)
    output_path = os.path.join(args.data_dir, args.output_folder)
    
    # create output data
    if not os.path.exists(output_path):
        os.mkdir(output_path)
        
    # load input 1
    file_list_1 = [listx for listx in os.listdir(input_path_1) if "json" in listx]
    
    query2negatives = {}
    for file in tqdm(file_list_1):
        file_1 = os.path.join(input_path_1, file)
        with open(file_1, "r", encoding="utf-8") as fi:
            for line in fi:
                data = json.loads(line)
                query = "_".join(str(ids) for ids in data["query"])
                negatives = data["negatives"]
                query2negatives[query] = negatives
                
    # load input 2 & write mix
    file_list_2 = [listx for listx in os.listdir(input_path_2) if "json" in listx]
    
    neg_num_list = []
    diff_num = 0
    for file in tqdm(file_list_2):
        file_2 = os.path.join(input_path_2, file)
        output_file = os.path.join(output_path, file)
        with open(file_2, "r", encoding="utf-8") as fi, \
            open(output_file, "w", encoding="utf-8") as fw:
            for line in fi:
                data = json.loads(line)
                query = data["query"]
                positives = data["positives"]
                qid = "_".join(str(ids) for ids in query)
                if qid in query2negatives:
                    negatives = data["negatives"] + query2negatives[qid]
                    neg_num_list.append(len(negatives))
                else:
                    negatives = data["negatives"]
                    diff_num += 1

                mix_example = {
                    'query': query,
                    'positives': positives,
                    'negatives': negatives,
                }
                fw.write(json.dumps(mix_example) + '\n')
               
    
    print("diff num = ", diff_num)
    print("combine neg num = ", np.mean(neg_num_list))
