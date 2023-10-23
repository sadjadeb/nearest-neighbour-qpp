import pickle
from collections import defaultdict
import os
import argparse
import json
from tqdm import tqdm

data_folder = 'data'
msmarco_folder = '../msmarco'


def main(target_metric, language_model):
    col_dic = defaultdict(list)
    collection_filepath = os.path.join(msmarco_folder, 'collection.tsv')
    with open(collection_filepath, 'r') as f:
        for line in f:
            doc_id, doc_text = line.strip().split('\t')
            col_dic[doc_id] = doc_text

    q_map_dic = {}
    queries_filepath = os.path.join(msmarco_folder, 'queries.train.judged.tsv')
    with open(queries_filepath, 'r') as f:
        for line in f:
            qid, qtext = line.strip().split('\t')
            q_map_dic[qid] = {'qtext': qtext}

    # a file that contains the performance of each query - produced by the script: extract_metrics_per_query.py
    metrics_filepath = os.path.join(data_folder, 'eval_per_query', 'runbm25anserini_evaluation-per-query.json')
    with open(metrics_filepath, 'r') as f:
        metrics = json.load(f)
        for qid in metrics.keys():
            q_map_dic[qid]['performance'] = metrics[qid][target_metric]

    # delete queries that do not have a performance
    for qid in list(q_map_dic.keys()):
        if 'performance' not in q_map_dic[qid].keys():
            del q_map_dic[qid]

    # add the first retrieved document by the query
    run_filepath = os.path.join(msmarco_folder, 'runbm25anserini')
    with open(run_filepath, 'r') as f:
        for line in tqdm(f):
            qid, _, doc_id, rank, _, _ = line.strip().split()
            if qid in q_map_dic.keys() and rank == '1':
                q_map_dic[qid]["doc_text"] = col_dic[doc_id]

    # a file that map a query to a similar query - produced by the script: find_most_similar_query.py
    matched_queries_filepath = os.path.join(data_folder, 'similar_queries', f'top_1_train-train_{language_model}_matched_queries.tsv')
    with open(matched_queries_filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines[1:]:
            qid, qtext, matched_qid, matched_qtext, rank = line.strip().split('\t')
            if qid in q_map_dic.keys():
                if matched_qid in q_map_dic.keys():
                    q_map_dic[qid]["matched_qtext"] = q_map_dic[matched_qid]["qtext"]
                    q_map_dic[qid]["matched_performance"] = str(int(q_map_dic[matched_qid]["performance"] * 100))
                else:
                    del q_map_dic[qid]

    # delete queries that do not have one of the above fields
    for qid in list(q_map_dic.keys()):
        if len(q_map_dic[qid].keys()) != 5:
            del q_map_dic[qid]

    output_path = os.path.join(data_folder, 'pkl_files', f'train_{language_model}_{target_metric.replace("@", "")}.pkl')
    with open(output_path, 'wb') as f:
        pickle.dump(q_map_dic, f, pickle.HIGHEST_PROTOCOL)
        print(f"Saved {len(q_map_dic)} queries to pkl file.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--target_metric", type=str, default="mrr@10")
    parser.add_argument("--language_model", type=str, default="microsoft/deberta-v3-base")
    args = parser.parse_args()

    main(args.target_metric, args.language_model.split("/")[-1])
