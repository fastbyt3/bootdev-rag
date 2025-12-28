import json
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
        bm25_search_results = self._bm25_search(query, limit * 500)
        sematic_search_results = self.semantic_search.search_chunks(query, limit * 500)

        normalized_bm25 = normalize([res["score"] for res in bm25_search_results])
        normalized_semantic = normalize(
            [res["score"] for res in sematic_search_results]
        )

        res_doc_map = {}

        for i in range(len(bm25_search_results)):
            bm_25_doc = bm25_search_results[i]["doc_id"]
            bm_25_score = normalized_bm25[i]
            if bm_25_doc in res_doc_map:
                res_doc_map[bm_25_doc]["bm_25_score"] = bm_25_score
            else:
                res_doc_map[bm_25_doc] = {
                    "id": bm_25_doc,
                    "title": bm25_search_results[i]["title"],
                    "bm_25_score": bm_25_score,
                }

            semantic_doc = sematic_search_results[i]["id"]
            semantic_score = normalized_semantic[i]
            if semantic_doc in res_doc_map:
                res_doc_map[semantic_doc]["semantic_score"] = bm_25_score
            else:
                res_doc_map[semantic_doc] = {
                    "id": semantic_doc,
                    "title": sematic_search_results[i]["title"],
                    "semantic_score": semantic_score,
                }

        for id in list(res_doc_map.keys()):
            if (
                "bm_25_score" not in res_doc_map[id]
                or "semantic_score" not in res_doc_map[id]
            ):
                del res_doc_map[id]
                continue
            res_doc_map[id]["hybrid_score"] = hybrid_score(
                res_doc_map[id]["bm_25_score"], res_doc_map[id]["semantic_score"], alpha
            )

        print(list(res_doc_map.items())[:10])
        return sorted(
            [item for item in res_doc_map.items()],
            key=lambda item: item[1]["hybrid_score"],
            reverse=True,
        )[:limit]

    def rrf_search(self, query, k, limit=10):
        bm25_results = self._bm25_search(query, limit * 500)
        semantic_res = self.semantic_search.search_chunks(query, limit * 500)

        docs = {}
        for i in range(len(bm25_results)):
            bm25_doc = bm25_results[i]
            sem_doc = semantic_res[i]

            if bm25_doc["doc_id"] not in docs:
                docs[bm25_doc["doc_id"]] = {
                    "doc_id": bm25_doc["doc_id"],
                    "title": bm25_doc["title"],
                    "bm25_rank": i,
                }
            else:
                docs[bm25_doc["doc_id"]]["bm25_rank"] = i

            if sem_doc["id"] not in docs:
                docs[sem_doc["id"]] = {
                    "doc_id": sem_doc["id"],
                    "title": sem_doc["title"],
                    "sem_rank": i,
                }
            else:
                docs[sem_doc["id"]]["sem_rank"] = i

        for id in list(docs.keys()):
            doc = docs[id]
            if "bm25_rank" not in doc or "sem_rank" not in doc:
                del docs[id]
                continue
            doc["rrf_score"] = rrf_score(doc["bm25_rank"], k) + rrf_score(
                doc["sem_rank"], k
            )

        return sorted(
            [
                {
                    "title": doc["title"],
                    "bm25_rank": doc["bm25_rank"],
                    "semantic_rank": doc["sem_rank"],
                    "rrf_score": doc["rrf_score"],
                }
                for doc in docs.values()
                if "rrf_score" in doc
            ],
            key=lambda doc: doc["rrf_score"],
            reverse=True,
        )[:limit]


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


def hybrid_score(bm25_score: float, semantic_score: float, alpha: float = 0.5) -> float:
    return alpha * bm25_score + (1 - alpha) * semantic_score


def rrf_score(rank, k=60):
    return 1 / (k + rank)


def weighted_search(query: str, alpha: float, limit: int):
    documents = json.load(open(MOVIES_DATA_FILE, "r"))["movies"]
    search = HybridSearch(documents)
    results = search.weighted_search(query, alpha, limit)
    print(results)

    for i, item in enumerate(results):
        _, res = item
        print(f"{i+1}. {res["title"]}")
        print(f"\tHybrid score: {res["hybrid_score"]:.3f}")
        print(f"\tBM25 score: {res["bm_25_score"]:.3f}")
        print(f"\tSemantic score: {res["semantic_score"]:.3f}")
        print("")


def rrf_search(query: str, k: float, limit: int):
    documents = json.load(open(MOVIES_DATA_FILE, "r"))["movies"]
    search = HybridSearch(documents)
    results = search.rrf_search(query, k, limit)
    print(results)

    for i, res in enumerate(results):
        print(f"{i+1}. {res["title"]}")
        print(f"\tRRF Score: {res["rrf_score"]:.3f}")
        print(f"\tBM25 rank: {res["bm25_rank"]}")
        print(f"\tSemantic rank: {res["semantic_rank"]}")
        print("")
