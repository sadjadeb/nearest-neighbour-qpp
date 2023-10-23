import random
import os
import json
from collections import defaultdict
from tqdm import tqdm
import math
from torch.utils.data import DataLoader
import torch
from sentence_transformers import CrossEncoder, InputExample
import gc

data_folder = 'data'
msmarco_folder = '../msmarco'
train_metric = 'map@20'
target_metric = 'mrr@10'
test_data = 'dev_small'
batch_size = 16
epoch_num = 1
device = 'cuda:1' if torch.cuda.is_available() else 'cpu'
model_name = 'microsoft/deberta-v3-base'
output_path = 'multiple'
num_samples = 10
sample_size = 100000

print('Reading the collection file...')
col_dic = defaultdict(list)
collection_filepath = os.path.join(msmarco_folder, 'collection.tsv')
with open(collection_filepath, 'r') as f:
    for line in f:
        doc_id, doc_text = line.strip().split('\t')
        col_dic[doc_id] = doc_text

# Read the lines from the file into a list
print('Reading the queries file...')
queries = {}
queries_filepath = os.path.join(msmarco_folder, 'queries.train.judged.tsv')
with open(queries_filepath, 'r') as f:
    for line in f:
        qid, qtext = line.strip().split('\t')
        queries[qid] = qtext

print('Reading the metrics file...')
metrics_filepath = os.path.join(data_folder, 'eval_per_query', 'runbm25anserini_evaluation-per-query.json')
with open(metrics_filepath, 'r') as f:
    metrics = json.load(f)

matched = {}
print('Reading the matched queries file...')
matched_queries_filepath = os.path.join(data_folder, 'similar_queries', f'top_1000_train-train_deberta-v3-base_matched_queries.tsv')
with open(matched_queries_filepath, 'r', encoding='utf-8') as f:
    lines = f.readlines()
    for line in lines[1:]:
        qid, qtext, matched_qid, matched_qtext, rank = line.strip().split('\t')
        if qid not in matched.keys():
            matched[qid] = []
        matched[qid].append(matched_qid)
    del lines

gc.collect()


def get_most_similar_query(qid, sample):
    for matched_qid in matched[qid]:
        if matched_qid in sample:
            return queries[matched_qid], metrics[matched_qid][train_metric]
    return None, None


def create_train_pkl(sample):
    q_map_dic = {}
    for qid in sample:
        q_map_dic[qid] = {'qtext': queries[qid]}

    for qid in list(q_map_dic.keys()):
        try:
            q_map_dic[qid]['performance'] = metrics[qid][train_metric]
        except KeyError:
            del q_map_dic[qid]

    # add the first retrieved document by the query
    run_filepath = os.path.join(msmarco_folder, 'runbm25anserini')
    with open(run_filepath, 'r') as f:
        for line in f:
            qid, _, doc_id, rank, _, _ = line.strip().split()
            if qid in q_map_dic.keys() and rank == '1':
                q_map_dic[qid]["doc_text"] = col_dic[doc_id]

    sample = list(q_map_dic.keys())
    for qid in sample:
        matched_qtext, matched_performance = get_most_similar_query(qid, sample)
        if matched_qtext is not None:
            q_map_dic[qid]["matched_qtext"] = matched_qtext
            q_map_dic[qid]["matched_performance"] = str(int(matched_performance * 100))
        else:
            del q_map_dic[qid]

    # delete queries that do not have one of the above fields
    for qid in list(q_map_dic.keys()):
        if len(q_map_dic[qid].keys()) != 5:
            del q_map_dic[qid]

    return q_map_dic


def train(q_dic_train, counter):
    model = CrossEncoder(model_name, num_labels=1, max_length=512, device=device)

    train_set = []
    for key in q_dic_train:
        q_text = q_dic_train[key]["qtext"]
        first_doc_text = q_dic_train[key]["doc_text"]
        actual_map = q_dic_train[key]["performance"]
        qp_text = q_dic_train[key]["matched_qtext"]
        qp_performance = q_dic_train[key]["matched_performance"]
        # concatenate the query, the query', the performance of query' and the first retrieved document using [SEP] token
        concatenated_text1 = q_text + " [SEP] " + qp_text
        concatenated_text2 = qp_performance + " [SEP] " + first_doc_text
        train_set.append(InputExample(texts=[concatenated_text1, concatenated_text2], label=actual_map))

    train_dataloader = DataLoader(train_set, shuffle=True, batch_size=batch_size)

    # Configure the training
    warmup_steps = math.ceil(len(train_dataloader) * epoch_num * 0.1)  # 10% of train data for warm-up

    # Train the model
    model.fit(train_dataloader=train_dataloader, epochs=epoch_num, warmup_steps=warmup_steps)

    model_path = os.path.join(output_path, f'IoQS_{train_metric}_{sample_size}_{counter}')
    os.makedirs(model_path, exist_ok=True)
    model.save(model_path)

    return model


def create_test_pkl(test_data):
    if test_data == "dev_small":
        test_data = "dev"

    # a file that map a query to a similar query - produced by the script: find_most_similar_query.py
    matched_queries_filepath = os.path.join(data_folder, 'similar_queries', f'top_1_train-{test_data}_all-MiniLM-L6-v2_matched_queries.tsv')
    q_map_dic = {}
    with open(matched_queries_filepath, 'r') as f:
        lines = f.readlines()
        for line in lines[1:]:
            qid, qtext, matched_qid, matched_qtext, rank = line.strip().split('\t')
            q_map_dic[qid] = {}
            q_map_dic[qid]["qtext"] = qtext
            q_map_dic[qid]["matched_qtext"] = matched_qtext
            q_map_dic[qid]["matched_performance"] = str(int(metrics[matched_qid][target_metric] * 100))

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

    return q_map_dic


def test(q_dic_test, trained_model, counter, train_set_length):
    sentences = []
    queries = []
    for key in q_dic_test:
        q_text = q_dic_test[key]["qtext"]
        first_doc_text = q_dic_test[key]["doc_text"]
        qp_text = q_dic_test[key]["matched_qtext"]
        qp_performance = q_dic_test[key]["matched_performance"]
        concatenated_text1 = q_text + " [SEP] " + qp_text
        concatenated_text2 = qp_performance + " [SEP] " + first_doc_text
        sentences.append([concatenated_text1, concatenated_text2])
        queries.append(key)

    scores = trained_model.predict(sentences)
    predicted = []
    with open(output_path + f'/results_{test_data}_{train_metric}_{target_metric}_{sample_size}_{counter}.txt', 'w') as f:
        f.write(f'{train_set_length}\n')
        for i in range(len(sentences)):
            predicted.append(float(scores[i]))
            f.write(queries[i] + '\t' + str(predicted[i]) + '\n')


def main():
    # Initialize an empty list to store the samples
    samples = []

    # Generate the random samples
    print('Generating random samples...')
    for _ in range(num_samples):
        # Randomly select sample_size lines without replacement
        sample = random.sample(queries.keys(), sample_size)
        samples.append(sample)

    counter = 0
    for sample in tqdm(samples, desc='samples'):
        q_dic_train = create_train_pkl(sample)
        trained_model = train(q_dic_train, counter)
        q_dic_test = create_test_pkl(test_data)
        test(q_dic_test, trained_model, counter, len(q_dic_train))
        counter += 1


if __name__ == '__main__':
    main()
