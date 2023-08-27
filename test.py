import pickle
from sentence_transformers.cross_encoder import CrossEncoder

trained_model = "output/tuned_model-ce_bert-base-uncased_epoch1_batch8_matched"

with open('data/pkl_files/dev_small_map.pkl', 'rb') as f:
    q_dic_test = pickle.load(f)

sentences = []
queries = []
for key in q_dic_test:
    q_text = q_dic_test[key]["qtext"]
    first_doc_text = q_dic_test[key]["doc_text"]
    qp_text = q_dic_test[key]["matched_qtext"]
    qp_performance = q_dic_test[key]["matched_performance"]
    sentences.append([q_text, qp_text, qp_performance, first_doc_text])
    queries.append(key)

model = CrossEncoder(trained_model, num_labels=1)
scores = model.predict(sentences)
predicted = []
with open(trained_model + '/results.txt', 'w') as f:
    for i in range(len(sentences)):
        predicted.append(float(scores[i]))
        f.write(queries[i] + '\t' + str(predicted[i]) + '\n')
