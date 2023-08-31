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
q_map_dic = {}
map_filepath = os.path.join(data_folder, 'eval_per_query', 'train_query_map_20.tsv')
with open(map_filepath, 'r') as f:
    for line in f:
        qid, qtext, performance = line.split('\t')
        q_map_dic[qid] = {}
        q_map_dic[qid]["qtext"] = qtext
        q_map_dic[qid]["performance"] = float(performance)

# run file including first retrieved documents per query
# qid<\t>doc_id<\t>rank
first_docs_filepath = os.path.join(data_folder, 'first_docs', 'bm25_first_docs_train.tsv')
with open(first_docs_filepath, 'r') as f:
    for line in f:
        qid, doc_id, rank = line.split('\t')
        if qid in q_map_dic.keys():
            q_map_dic[qid]["doc_text"] = col_dic[doc_id]

# a file that map a query to a positive query
# qid<\t>pos_qid
matched_queries_filepath = os.path.join(data_folder, 'top_1_train-train_matched_queries.tsv')
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

with open('data/pkl_files/train_map.pkl', 'wb') as f:
    pickle.dump(q_map_dic, f, pickle.HIGHEST_PROTOCOL)
    print(f"Saved {len(q_map_dic)} queries to pkl file.")
