import argparse
import math


def min_max_normalize(score, smin, smax):
    normalized_score = (score - smin) / (smax - smin)
    return normalized_score


def standard_score_normalize(score, mean, std_dev):
    normalized_score = (score - mean) / std_dev
    return normalized_score


def sum_normalize(score, sum):
    normalized_score = score / sum
    return normalized_score


def normalize_scores(input_filename, output_filename, method):
    # Read data from the input file
    data = []
    with open(input_filename, 'r', encoding='utf-8') as f:
        for line in f:
            qid, q0, doc_id, rank, score, model_name = line.strip().split()
            data.append((qid, q0, doc_id, rank, float(score), model_name))

    # Find min and max scores
    scores = [score for _, _, _, _, score, _ in data]
    smin = min(scores)
    smax = max(scores)
    sum_scores = sum(scores)
    mean_score = sum(scores) / len(scores)
    std_dev_score = math.sqrt(sum((score - mean_score) ** 2 for score in scores) / len(scores))
    print(f"Min score: {smin}")
    print(f"Max score: {smax}")
    print(f"Mean score: {mean_score}")
    print(f"Std dev score: {std_dev_score}")

    # Normalize scores and write to the output file
    with open(output_filename, 'w', encoding='utf-8') as f:
        for qid, q0, doc_id, rank, score, model_name in data:
            if method == 'min_max':
                normalized_score = min_max_normalize(score, smin, smax)
            elif method == 'standard_score':
                normalized_score = standard_score_normalize(score, mean_score, std_dev_score)
            elif method == 'sum':
                normalized_score = sum_normalize(score, sum_scores)
            else:
                raise ValueError(f"Invalid method: {method}")
            normalized_score = int(normalized_score * 100)
            f.write(f"{qid} {q0} {doc_id} {rank} {normalized_score} {model_name}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--run_file', type=str, required=True)
    parser.add_argument('--method', type=str, choices=['min_max', 'standard_score', 'sum'], default='min_max')

    args = parser.parse_args()
    normalize_scores(args.run_file, args.run_file + '.normalized', args.method)
