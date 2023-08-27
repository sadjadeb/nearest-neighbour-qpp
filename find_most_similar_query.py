import faiss
from sentence_transformers import SentenceTransformer
import os
import torch
import argparse
import time


def find_similar_queries(args):
    model_name = args.model_name
    base_queries_filepath = args.base_queries
    output_path = args.output_path

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    model = SentenceTransformer(model_name)
    embedding_dimension_size = model.get_sentence_embedding_dimension()

    base_queries = {}
    queries_list = []
    with open(base_queries_filepath, 'r', encoding='utf-8') as fIn:
        for line in fIn:
            qid, query = line.strip().split("\t")
            base_queries[qid] = query
            queries_list.append(query)

    print("Number of Queries to be indexed:", len(queries_list))

    queries_embeddings = model.encode(queries_list, convert_to_tensor=True, show_progress_bar=True, batch_size=128)
    if args.save_embeddings:
        torch.save(queries_embeddings, output_path + '/queries_tensor.pt')

    index = faiss.IndexFlatL2(embedding_dimension_size)
    print(index.is_trained)
    index.add(queries_embeddings.detach().cpu().numpy())
    print(index.ntotal)
    if args.save_index:
        faiss.write_index(index, output_path + '/faiss.index')

    target_queries_filepath = args.target_queries
    output_path = args.output_path
    top_k = args.hits + 1

    target_queries = {}
    queries_list = []
    with open(target_queries_filepath, 'r', encoding='utf-8') as fIn:
        for line in fIn:
            qid, query = line.strip().split("\t")
            target_queries[qid] = query
            queries_list.append(query.strip())

    xq = model.encode(queries_list)

    start_time = time.time()
    D, I = index.search(xq, top_k)
    print(f'Search time: {time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time))}')

    base_queries_name = base_queries_filepath.split('/')[-1].split('.')[1]
    target_queries_name = target_queries_filepath.split('/')[-1].split('.')[1]

    with open(output_path + f'/top_{top_k - 1}_{base_queries_name}-{target_queries_name}_matched_queries.tsv', 'w', encoding='utf-8') as fOut:
        t_qids = list(target_queries.keys())
        b_qids = list(base_queries.keys())
        fOut.write(f'qid\tquery\tmatched_qid\tmatched_query\trank\n')
        if base_queries_filepath == target_queries_filepath:
            for qid in range(len(I)):
                for rank in range(1, top_k):
                    fOut.write(f'{t_qids[qid]}\t{target_queries[t_qids[qid]]}\t{b_qids[I[qid][rank]]}\t{base_queries[b_qids[I[qid][rank]]]}\t{rank}\n')
        else:
            for qid in range(len(I)):
                for rank in range(top_k - 1):
                    fOut.write(f'{t_qids[qid]}\t{target_queries[t_qids[qid]]}\t{b_qids[I[qid][rank]]}\t{base_queries[b_qids[I[qid][rank]]]}\t{rank + 1}\n')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str,
                        default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument('--base_queries', type=str)
    parser.add_argument('--target_queries', type=str)
    parser.add_argument('--save_embeddings', type=bool, default=False)
    parser.add_argument('--save_index', type=bool, default=True)
    parser.add_argument('--output_path', type=str)  # path to output file
    parser.add_argument('--hits', type=int)  # number of top-k most similar matched queries to be selected
    args = parser.parse_args()

    find_similar_queries(args)
