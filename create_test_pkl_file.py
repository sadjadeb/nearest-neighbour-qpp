import pickle
from collections import defaultdict
import os
import json
import argparse
from tqdm import tqdm

data_folder = 'data'
msmarco_folder = '../msmarco'


def main(test_data, target_metric, language_model):
    if test_data == "dev_small":
        test_data = "dev"

    if test_data == "trec-dl-2021":
        collection_name = "collection-v2.tsv"
    else:
        collection_name = "collection.tsv"
    collection_filepath = os.path.join(msmarco_folder, collection_name)
    col_dic = defaultdict(list)
    with open(collection_filepath, 'r') as f:
        for line in f:
            doc_id, doc_text = line.strip().split('\t')
            col_dic[doc_id] = doc_text

    # a file that contains the performance of each query - produced by the script: extract_metrics_per_query.py
    train_performance = {}
    metrics_filepath = os.path.join(data_folder, 'eval_per_query', 'runbm25anserini_evaluation-per-query.json')
    with open(metrics_filepath, 'r') as f:
        metrics = json.load(f)
        for qid in metrics.keys():
            train_performance[qid] = metrics[qid][target_metric]

    # a file that map a query to a similar query - produced by the script: find_most_similar_query.py
    matched_queries_filepath = os.path.join(data_folder, 'similar_queries', f'top_1_train-{test_data}_{language_model}_matched_queries.tsv')
    q_map_dic = {}
    with open(matched_queries_filepath, 'r') as f:
        lines = f.readlines()
        for line in lines[1:]:
            qid, qtext, matched_qid, matched_qtext, rank = line.strip().split('\t')
            q_map_dic[qid] = {}
            q_map_dic[qid]["qtext"] = qtext
            q_map_dic[qid]["matched_qtext"] = matched_qtext
            q_map_dic[qid]["matched_performance"] = str(int(train_performance[matched_qid] * 100))

    # add the first retrieved document by the query
    run_filepath = os.path.join(msmarco_folder, f'runbm25anserini.{test_data}')
    with open(run_filepath, 'r') as f:
        for line in tqdm(f):
            qid, _, doc_id, rank, _, _ = line.strip().split()
            if qid in q_map_dic.keys() and rank == '1':
                q_map_dic[qid]["doc_text"] = col_dic[doc_id]

    # delete queries that do not have one of the above fields
    for qid in list(q_map_dic.keys()):
        if len(q_map_dic[qid].keys()) != 4:
            del q_map_dic[qid]

    output_path = os.path.join(data_folder, 'pkl_files', f'{test_data}_{language_model}_{target_metric.replace("@", "")}.pkl')
    with open(output_path, 'wb') as f:
        pickle.dump(q_map_dic, f, pickle.HIGHEST_PROTOCOL)
        print(f"Saved {len(q_map_dic)} queries to pkl file.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_data", type=str, default="dev_small", choices=["dev_small", "trec-dl-2019", "trec-dl-2020", "trec-dl-2021"])
    parser.add_argument("--target_metric", type=str, default="mrr@10")
    parser.add_argument("--language_model", type=str, default="microsoft/deberta-v3-base")
    args = parser.parse_args()

    main(args.test_data, args.target_metric, args.language_model.split("/")[-1])
