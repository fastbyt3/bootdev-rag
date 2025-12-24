import os

from constants import MOVIES_DATA_FILE
from lib.chunked_semantic_search import ChunkedSemanticSearch
from lib.inverted_index import InvertedIndex


class HybridSearch:
    def __init__(self, documents):
        self.documents = documents
        self.semantic_search = ChunkedSemanticSearch()
        self.semantic_search.load_or_create_chunk_embeddings(documents)

        self.idx = InvertedIndex()
        if not os.path.exists(self.idx.index_cache_file):
            self.idx.build(MOVIES_DATA_FILE)
            self.idx.save()

    def _bm25_search(self, query, limit):
        self.idx.load()
        return self.idx.bm25_search(query, limit)

    def weighted_search(self, query, alpha, limit=5):
        raise NotImplementedError("Weighted hybrid search is not implemented yet.")

    def rrf_search(self, query, k, limit=10):
        raise NotImplementedError("RRF hybrid search is not implemented yet.")


def normalize(scores: list[float]) -> list[float]:
    if len(scores) == 0:
        return []

    def normalize_formula(score, min_score, max_score):
        return (score - min_score) / (max_score - min_score)

    min_score = min(scores)
    max_score = max(scores)

    if min_score == max_score:
        return [1.0] * len(scores)

    return [normalize_formula(score, min_score, max_score) for score in scores]
