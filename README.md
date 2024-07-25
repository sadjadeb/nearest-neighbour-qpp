# Pre-Retrieval NN-QPP: Estimating Query Performance based on Nearest Neighbor Sampling
This repository contains the code and resources for our proposed pre-retrieval Query Performance Prediction (QPP) method that leverages nearest neighbors retrieval strategy for predicting the performance of the input query. To do so, we propose to maintain a Querystore where queries with known performances are indexed and sampled at runtime if they are the nearest neighbors of the input query. The performance of the sampled queries are used to estimate the possible performance of the new query. The framework of our propsoed Nearest Neighbor QPP (NN-QPP) method is shown below:
<p align="center">
  <img src="https://github.com/Narabzad/NN-QPP/blob/main/architecture.png" width="800" height="400">
</p>

## Performance Comparison with Baselines
The table below shows Pearson Rho, kendall Tau, and Spearman correlation of different baselines as well as our proposed NN-QPP method over four different datasets.

<table>
<thead>
  <tr>
    <th rowspan="2">QPP Method</th>
    <th colspan="3">MS MARCO Dev small (6980 queries)</th>
    <th colspan="3">TREC DL 2019 (43 Queries)</th>
    <th colspan="3">TREC DL 2020 (53 Queries)</th>
    <th colspan="3">DL Hard (50 Queries)</th>
  </tr>
  <tr>
    <th>Pearson Rho</th>
    <th>kendall Tau</th>
    <th>Spearman</th>
    <th>Pearson Rho</th>
    <th>kendall Tau</th>
    <th>Spearman</th>
    <th>Pearson Rho</th>
    <th>kendall Tau</th>
    <th>Spearman</th>
    <th>Pearson Rho</th>
    <th>kendall Tau</th>
    <th>Spearman</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td>SCS</td>
    <td>0.021</td>
    <td>0.058</td>
    <td>0.085</td>
    <td>0.471</td>
    <td>0.262</td>
    <td>0.354</td>
    <td>0.447</td>
    <td>0.310</td>
    <td>0.448</td>
    <td>0.247</td>
    <td>0.159</td>
    <td>0.240</td>
  </tr>
  <tr>
    <td>P_Clarity</td>
    <td>0.052</td>
    <td>0.007</td>
    <td>0.009</td>
    <td>0.109</td>
    <td>0.119</td>
    <td>0.139</td>
    <td>0.069</td>
    <td>0.052</td>
    <td>0.063</td>
    <td>0.095</td>
    <td>0.209</td>
    <td>0.272</td>
  </tr>
  <tr>
    <td>VAR</td>
    <td>0.067</td>
    <td>0.081</td>
    <td>0.119</td>
    <td>0.290</td>
    <td>0.141</td>
    <td>0.187</td>
    <td>0.047</td>
    <td>0.051</td>
    <td>0.063</td>
    <td>0.023</td>
    <td>0.014</td>
    <td>0.001</td>
  </tr>
  <tr>
    <td>PMI</td>
    <td>0.030</td>
    <td>0.033</td>
    <td>0.048</td>
    <td>0.155</td>
    <td>0.065</td>
    <td>0.079</td>
    <td>0.021</td>
    <td>0.012</td>
    <td>0.003</td>
    <td>0.093</td>
    <td>0.027</td>
    <td>0.042</td>
  </tr>
  <tr>
    <td>IDF</td>
    <td>0.117</td>
    <td>0.138</td>
    <td>0.200</td>
    <td>0.440</td>
    <td>0.276</td>
    <td>0.389</td>
    <td>0.413</td>
    <td>0.236</td>
    <td>0.345</td>
    <td>0.200</td>
    <td>0.197</td>
    <td>0.275</td>
  </tr>
  <tr>
    <td>SCQ</td>
    <td>0.029</td>
    <td>0.022</td>
    <td>0.032</td>
    <td>0.395</td>
    <td>0.114</td>
    <td>0.157</td>
    <td>0.193</td>
    <td>0.005</td>
    <td>0.004</td>
    <td>0.335</td>
    <td>0.106</td>
    <td>0.152</td>
  </tr>
  <tr>
    <td>ICTF</td>
    <td>0.105</td>
    <td>0.136</td>
    <td>0.198</td>
    <td>0.435</td>
    <td>0.259</td>
    <td>0.365</td>
    <td>0.409</td>
    <td>0.236</td>
    <td>0.348</td>
    <td>0.192</td>
    <td>0.195</td>
    <td>0.272</td>
  </tr>
  <tr>
    <td>DC</td>
    <td>0.071 </td>
    <td>0.044</td>
    <td>0.065</td>
    <td>0.132</td>
    <td>0.083</td>
    <td>0.092</td>
    <td>0.1001 </td>
    <td>0.1175</td>
    <td>0.14913</td>
    <td>0.155</td>
    <td>0.091</td>
    <td>0.115</td>
  </tr>
  <tr>
    <td>CC</td>
    <td>0.085</td>
    <td>0.066</td>
    <td>0.076</td>
    <td>0.079 </td>
    <td>0.068</td>
    <td>0.023</td>
    <td>0.172</td>
    <td>0.065</td>
    <td>0.089</td>
    <td>0.155</td>
    <td>0.093</td>
    <td>0.111</td>
  </tr>
  <tr>
    <td>IEF</td>
    <td>0.110</td>
    <td>0.090</td>
    <td>0.118</td>
    <td>0.140</td>
    <td>0.090</td>
    <td>0.134</td>
    <td>0.110</td>
    <td>0.025</td>
    <td>0.037</td>
    <td>0.018</td>
    <td>0.071</td>
    <td>0.139</td>
  </tr>
  <tr>
    <td>MRL</td>
    <td>0.022</td>
    <td>0.046</td>
    <td>0.067</td>
    <td>0.176</td>
    <td>0.079</td>
    <td>0.140</td>
    <td>0.093</td>
    <td>0.078</td>
    <td>0.117</td>
    <td>-0.046</td>
    <td>0.052</td>
    <td>0.038</td>
  </tr>
  <tr>
    <td>NN-QPP</td>
    <td>0.219</td>
    <td>0.214</td>
    <td>0.309</td>
    <td>0.483</td>
    <td>0.349</td>
    <td>0.508</td>
    <td>0.452</td>
    <td>0.319</td>
    <td>0.457</td>
    <td>0.364</td>
    <td>0.234</td>
    <td>0.340</td>
  </tr>
</tbody>
</table>

## Ablation Study
The performance of NN-QPP may be impacted by the choice of (1) the base language model that is used for creating the Querystore, (2) the number of nearest neighbor samples that are retrieved per query during inference time, and (3) the size of Querystore used for finding the nearest neighbor samples. As such, we investigate their impact on the overall performance of the model. For this purpose, we adopt three different large language models, namely (1) [all-mpnet-base-v2](https://huggingface.co/sentence-transformers/all-mpnet-base-v2), (2) [all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) and (3) [paraphrase-MiniLM-v2](https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2) and develop the Querystore independently for each of them and measure the performance of NN-QPP. In addition, we sample queries from the Querystore based on k = {1,3, 5, 7, 9, 10} over all the four datasets. The figures include performance based on Kendall Tau, Pearson Rho, and Spearman correlations.
<p align="center">
  <img src="https://github.com/Narabzad/NN-QPP/blob/main/Diagrams.jpg">
</p>
In addition, we explore the impact of Querystore size on the performance of NN-QPP. To accomplish this, we employed a random sampling approach to select various percentages of queries from the pool of 500k MS MARCO queries. For each subset of queries, we construct distinct versions of the Querystore using the paraphrase-MiniLM-v2 language model. Subsequently, we evaluate the NN-QPP method on the MS MARCO Dev query dataset, utilizing the top-10 nearest neighbors sampled from each Querystore. The outcomes of these evaluations are presented in the Table below.
<div align="center">
<table>
<thead>
  <tr>
    <th>Percentage of Queries </th>
    <th>Pearson</th>
    <th>Kendall</th>
    <th>Spearman</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td>50%</td>
    <td>0.200</td>
    <td>0.191</td>
    <td>0.278</td>
  </tr>
  <tr>
    <td>60%</td>
    <td>0.200</td>
    <td>0.197</td>
    <td>0.286</td>
  </tr>
  <tr>
    <td>70%</td>
    <td>0.196</td>
    <td>0.199</td>
    <td>0.290</td>
  </tr>
  <tr>
    <td>80%</td>
    <td>0.216</td>
    <td>0.209</td>
    <td>0.302</td>
  </tr>
  <tr>
    <td>90%</td>
    <td>0.215</td>
    <td>0.207</td>
    <td>0.299</td>
  </tr>
  <tr>
    <td>100%</td>
    <td>0.219</td>
    <td>0.214</td>
    <td>0.309</td>
  </tr>
</tbody>
</table>
</div>

## Usage
##### In order to predict the performance of a set of target queries, you can follow the process below:
1- First calculate the performance of QueryStore queries using [QueryStorePerformanceCalculator.py](https://github.com/Narabzad/NN-QPP/blob/main/code/QueryStorePerformanceCalculator.py). This code receives a set of queries and calculate their performance (i.e. MAP@1000) through [anserini](https://github.com/castorini/anserini) toolkit.
```
python QueryStorePerformanceCalculator.py\
     -queries path to queries (TSV format) \
     -anserini path to anserini \
     -index path collection index \
     -qrels path to qrels \
     -nproc number of CPUs \
     -experiment_dir experiment folder \
     -queries_chunk_size chunk_size to split queries \
     -hits number of docs to retrieve for those queries and caluclate performance based on
```
MAP@1000 score of MS MARCO queries that were used to build the QueryStore are uploaded as a pickle file named [QueryStore_queries_MAP@1000.pkl](https://github.com/Narabzad/NN-QPP/blob/main/QueryStore_queries_MAP%401000.pkl). <br>
2- In order to find the most similar queries from the QueryStore and retreived the most similar queries during the inference, we need to first index the QueryStore queries. This can be done using [encode_queries.py]() as below:
```
python encode_queries.py\
     -model model we want to create embeddings with (i.e. sentence-transformers/all-MiniLM-L6-v2) \
     -queries path to queries we want to index (TSV format) \
     -output path to output folder 
```
3- During inferene, we can find the top_k most similar queries to a set of target queries from the QueryStore using the [find_most_similar_queries.py](https://github.com/Narabzad/NN-QPP/blob/main/code/find_most_similar_queries.py) script as below:
```
python find_most_similar_queries.py\
     -model model we want to create embeddings with for target queries (i.e. sentence-transformers/all-MiniLM-L6-v2) \
     -faiss_index path the index of QueryStore queries \
     -target_queries_path path to target_queries \
     -hits  #number of top-k most similar matched queries to be selected
```
4- Finally, having the top-k most similar queries for each of the target queries, we can calculate its performance by calculating the average of performance over retreived queries performance using [query_performance_predictor.py](https://github.com/Narabzad/NN-QPP/blob/main/code/query_performance_predictor.py) as follows:
```
python query_performance_predictor.py\
     -top_matched_queries path to top-k matched queries from QueryStore for target queries \
     -QueryStore_queries path to QueryStor queries TSV format \
     -QueryStore_queries_performance #path to the pickle file containing the MAP@1000 of QueryStor queries (QueryStore_queries_MAP@1000.pkl) \
     -output  path to output
```


# Post-Retrieval NN-QPP:: Estimating Query Performance Through Rich Contextualized Query Representations

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

