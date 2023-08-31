import pickle
from sentence_transformers.cross_encoder import CrossEncoder

model_name = 'bert-base-uncased'
trained_model = f"output/QPP_{model_name}_matched"

with open('data/pkl_files/dev_small_map.pkl', 'rb') as f:
    q_dic_test = pickle.load(f)

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

model = CrossEncoder(trained_model, num_labels=1)
scores = model.predict(sentences)
predicted = []
with open(trained_model + '/results.txt', 'w') as f:
    for i in range(len(sentences)):
        predicted.append(float(scores[i]))
        f.write(queries[i] + '\t' + str(predicted[i]) + '\n')
