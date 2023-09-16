import pickle
from sentence_transformers import SentenceTransformer, util

model_name = 'bert-base-uncased'
trained_model = f"output/QPP_bi_{model_name.split('/')[-1]}_matched"

with open('data/pkl_files/dev_all-MiniLM-L6-v2_mrr20.pkl', 'rb') as f:
    q_dic_test = pickle.load(f)

sentences1 = []
sentences2 = []
queries = []
for key in q_dic_test:
    q_text = q_dic_test[key]["qtext"]
    first_doc_text = q_dic_test[key]["doc_text"]
    qp_text = q_dic_test[key]["matched_qtext"]
    qp_performance = q_dic_test[key]["matched_performance"]
    concatenated_text1 = q_text + " [SEP] " + qp_text
    concatenated_text2 = qp_performance + " [SEP] " + first_doc_text
    sentences1.append(concatenated_text1)
    sentences2.append(concatenated_text2)
    queries.append(key)

model = SentenceTransformer(trained_model)
embeddings1 = model.encode(sentences1, convert_to_tensor=True)
embeddings2 = model.encode(sentences2, convert_to_tensor=True)
scores = util.pytorch_cos_sim(embeddings1, embeddings2)
with open(trained_model + '/results.txt', 'w') as f:
    for i in range(len(sentences1)):
        f.write(queries[i] + '\t' + str(float(scores[i][i])) + '\n')
