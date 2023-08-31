import pickle
from collections import defaultdict
import os

data_folder = 'data'

col_dic = defaultdict(list)
collection_filepath = os.path.join(data_folder, 'collection.tsv')
with open(collection_filepath, 'r') as f:
    for line in f:
        doc_id, doc_text = line.strip().split('\t')
        col_dic[doc_id] = doc_text

# path to evaluation metric per query
# qid<\t>query<\t>metric_value
train_performance = {}
map_filepath = os.path.join(data_folder, 'eval_per_query', 'train_query_map_20.tsv')
with open(map_filepath, 'r') as f:
    for line in f:
        qid, qtext, performance = line.strip().split('\t')
        train_performance[qid] = float(performance)

q_map_dic = {}
queries_filepath = os.path.join(data_folder, 'top_1_train-dev_matched_queries.tsv')
with open(queries_filepath, 'r') as f:
    lines = f.readlines()
    for line in lines[1:]:
        qid, qtext, matched_qid, matched_qtext, rank = line.strip().split('\t')
        q_map_dic[qid] = {}
        q_map_dic[qid]["qtext"] = qtext
        q_map_dic[qid]["matched_qtext"] = matched_qtext
        q_map_dic[qid]["matched_performance"] = str(int(train_performance[matched_qid] * 100))

first_docs_filepath = os.path.join(data_folder, 'run', 'bm25_first_docs_dev.tsv')
with open(first_docs_filepath, 'r') as f:
    lines = f.readlines()
    for line in lines:
        qid, doc_id, rank = line.split('\t')
        if qid in q_map_dic.keys():
            q_map_dic[qid]["doc_text"] = col_dic[doc_id]

with open('data/pkl_files/dev_small_map.pkl', 'wb') as f:
    pickle.dump(q_map_dic, f, pickle.HIGHEST_PROTOCOL)
    print(f"Saved {len(q_map_dic)} queries to pkl file.")
