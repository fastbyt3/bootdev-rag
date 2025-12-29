import argparse
import json

from constants import GOLDEN_DATASET_PATH
from lib.hybrid_search import rrf_search


def precision_at_k(
    retrieved_docs: list[str], relevant_docs: set[str], k: int = 5
) -> float:
    top_k = retrieved_docs[:k]
    relevant_count = 0
    for doc in top_k:
        if doc in relevant_docs:
            relevant_count += 1
    return relevant_count / k


def recall_at_k(
    retrieved_docs: list[str], relevant_docs: set[str], k: int = 5
) -> float:
    print("Retrieved:", retrieved_docs[:k])
    print("Relevant:", relevant_docs)
    print("Relevant retrieved:", [d for d in retrieved_docs[:k] if d in relevant_docs])
    top_k = retrieved_docs[:k]
    relevant_retrieved = sum([doc in relevant_docs for doc in top_k])
    total_relevant = len(relevant_docs)
    return relevant_retrieved / total_relevant


def main():
    parser = argparse.ArgumentParser(description="Search Evaluation CLI")
    parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Number of results to evaluate (k for precision@k, recall@k)",
    )

    args = parser.parse_args()
    limit = args.limit

    # run evaluation logic here
    test_cases = json.load(open(GOLDEN_DATASET_PATH, "r"))["test_cases"]

    for tc in test_cases:
        query = tc["query"]
        expected_docs = set(tc["relevant_docs"])

        rrf_search_results = rrf_search(query, k=6, limit=limit)
        assert rrf_search_results, "RRF Search results seems to be of None type"

        print(rrf_search_results)
        retrieved_titles = [doc["title"] for doc in rrf_search_results]

        precision = precision_at_k(retrieved_titles, expected_docs, limit)
        recall = recall_at_k(retrieved_titles, expected_docs, limit)

        print(f"- Query: {query}")
        print(f"\t- Precision@{limit}: {precision:.4f}")
        print(f"\t- Recall@{limit}: {recall:.4f}")
        print(f"\t- Retrieved: {", ".join(retrieved_titles)}")
        print(f"\t- Relevant: {", ".join(expected_docs)}")
        print("")


if __name__ == "__main__":
    main()
