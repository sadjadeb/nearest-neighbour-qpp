from sentence_transformers.cross_encoder import CrossEncoder
from sentence_transformers import InputExample
from torch.utils.data import DataLoader
import pickle
import math
import os

batch_size = 8
epoch_num = 1
model_name = 'bert-base-uncased'
model_path = f"output/tuned_model-ce_{model_name}_epoch{epoch_num}_batch{batch_size}_matched"

os.makedirs(model_path, exist_ok=True)

model = CrossEncoder(model_name, num_labels=1)

with open('data/pkl_files/train_map.pkl', 'rb') as f:
    q_dic_train = pickle.load(f)

train_set = []
for key in q_dic_train:
    q_text = q_dic_train[key]["qtext"]
    first_doc_text = q_dic_train[key]["doc_text"]
    actual_map = q_dic_train[key]["performance"]
    qp_text = q_dic_train[key]["matched_qtext"]
    qp_performance = q_dic_train[key]["matched_performance"]
    train_set.append(InputExample(texts=[q_text, qp_text, qp_performance, first_doc_text], label=actual_map))

train_dataloader = DataLoader(train_set, shuffle=True, batch_size=batch_size)

# Configure the training
warmup_steps = math.ceil(len(train_dataloader) * epoch_num * 0.1)  # 10% of train data for warm-up

# Train the model
model.fit(train_dataloader=train_dataloader,
          epochs=epoch_num,
          warmup_steps=warmup_steps,
          output_path=model_path)
model.save(model_path)
