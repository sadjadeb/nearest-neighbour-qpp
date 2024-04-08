# post-nn-qpp : Estimating Query Performance Through Rich Contextualized Query Representations

## Introduction
The state-of-the-art query performance prediction methods rely on the fine-tuning of contextual language models to estimate retrieval effectiveness on a per-query basis. Our work in this paper builds on this strong foundation and proposes to learn rich query representations by learning the interactions between the query and two important contextual information, namely the set of documents retrieved by that query, and the set of similar historical queries with known retrieval effectiveness. We propose that such contextualized query representations can be more accurate estimators of query performance as they embed the performance of past similar queries and the semantics of the documents retrieved by the query. We perform extensive experiments on the MSMARCO collection and its accompanying query sets including MSMARCO Dev set and TREC Deep Learning tracks of 2019, 2020, 2021,
and DL-Hard. Our experiments reveal that our proposed method shows robust and effective performance compared to state-of-the-art baselines.

## Running the code
first, you need to clone the repository:
```
git clone https://github.com/sadjadeb/nearest-neighbour-qpp.git
```
Then, you need to create a virtual environment and install the requirements:
```
cd nearest-neighbour-qpp/
sudo apt-get install virtualenv
virtualenv venv
source venv/bin/activate
pip install -r requirements.txt
```
Then, you need to download the data:
```
bash download_data.sh
```

### Prepare the data
To create a dictionary which maps each query to its actual performance by BM25 (i.e. MRR@10), you need to run the following command:
```
python extract_metrics_per_query.py --run /path/to/run/file --qrels /path/to/qrels/file --qrels /path/to/qrels/file
```
It will create a file named `run-file-name_evaluation-per-query.json` in the `data/eval_per_query` directory.

Then you need to create a file which contains the most similar query from train-set(a.k.a. historical queries with known retrieval effectiveness) to each query. To do so, you need to run the following command:
```
python find_most_similar_query.py --base_queries /path/to/train-set/queries --base_queries /path/to/train-set/queries --base_queries /path/to/train-set/queries --target_queries /path/to/desired/queries --target_queries /path/to/desired/queries --target_queries /path/to/desired/queries --model_name /name/of/the/model --model_name /name/of/the/model --model_name /name/of/the/language/model --hits /number/of/hits --hits /number/of/hits --hits /number/of/hits
```

Finally, to gather all data in a file to make it easier to load the data, you need to run the following commands:
```
python create_train_pkl_file.py
python create_test_pkl_file.py
```

### Training
To train the model, you need to run the following command:
```
python train.py
```
You can change the hyperparameters of the model by changing the values in the lines 9-12 of the `train.py` file.

### Testing
To test the model, you need to run the following command:
```
python test.py
```

### Evaluation
To evaluate the model, you need to run the following command:
```
python evaluation.py --actual /path/to/actual/performance/file --predicted /path/to/predicted/performance/file --target_metric /target/metric
```

