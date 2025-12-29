import json
import logging
import os

from config import Config
from constants import MOVIES_DATA_FILE
from google import genai
from lib.chunked_semantic_search import ChunkedSemanticSearch
from lib.inverted_index import InvertedIndex
from lib.prompts import (
    batch_rerank_prompt,
    expand_query_prompt,
    individual_rerank_prompt,
    rewrite_query_prompt,
    spell_check_prompt,
)
from sentence_transformers import CrossEncoder

logger = logging.getLogger(__name__)


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
                    "description": bm25_doc["description"],
                    "bm25_rank": i,
                }
            else:
                docs[bm25_doc["doc_id"]]["bm25_rank"] = i

            if sem_doc["id"] not in docs:
                docs[sem_doc["id"]] = {
                    "doc_id": sem_doc["id"],
                    "title": sem_doc["title"],
                    "description": sem_doc["description"],
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
                    "description": doc["description"],
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


def rrf_search(
    query: str,
    k: float,
    limit: int,
    enhance: str | None = None,
    rerank_method: str | None = None,
) -> list[dict]:
    if rerank_method in ["individual", "batch", "cross_encoder"]:
        limit = limit * 5

    if enhance:
        client = get_llm_client()

        match enhance:
            case "spell":
                response = client.models.generate_content(
                    model="gemini-3-flash-preview",
                    contents=spell_check_prompt(query),
                )
                assert response.text, "Failed to get text from LLM response"
                print(f"Enhanced query ({enhance}): '{query}' -> '{response.text}'\n")
                query = response.text
            case "rewrite":
                response = client.models.generate_content(
                    model="gemini-3-flash-preview", contents=rewrite_query_prompt(query)
                )
                assert response.text, "Failed to get text from LLM response"
                print(f"Enhanced query ({enhance}): '{query}' -> '{response.text}'\n")
                query = response.text
            case "expand":
                response = client.models.generate_content(
                    model="gemini-3-flash-preview", contents=expand_query_prompt(query)
                )
                assert response.text, "Failed to get text from LLM response"
                print(f"Enhanced query ({enhance}): '{query}' -> '{response.text}'\n")
                query = response.text

    documents = json.load(open(MOVIES_DATA_FILE, "r"))["movies"]
    search = HybridSearch(documents)
    results = search.rrf_search(query, k, limit)

    if not rerank_method:
        for i, res in enumerate(results):
            print(f"{i+1}. {res["title"]}")
            print(f"\tRRF Score: {res["rrf_score"]:.3f}")
            print(f"\tBM25 rank: {res["bm25_rank"]}")
            print(f"\tSemantic rank: {res["semantic_rank"]}")
            print("")
        return results

    if rerank_method == "individual":
        client = get_llm_client()
        for doc in results:
            response = client.models.generate_content(
                model="gemini-3-flash-preview",
                contents=individual_rerank_prompt(query, doc),
            )

            logger.debug(
                f"LLM response generated for {doc}. Response = {response.text}"
            )
            assert response.text, "LLM Response for re-ranking was empty"
            assert (
                response.text.isdigit()
            ), "LLM Response for re-ranking was not a valid number"

            doc["reranked_score"] = int(response.text)

        sorted_results = sorted(
            results,
            key=lambda doc: doc["reranked_score"],
            reverse=True,
        )[: int(limit / 5)]

        for i, res in enumerate(sorted_results):
            print(f"{i+1}. {res["title"]}")
            print(f"\tRerank Score: {res["reranked_score"]:.3f}/10")
            print(f"\tRRF Score: {res["rrf_score"]:.3f}")
            print(f"\tBM25 rank: {res["bm25_rank"]}")
            print(f"\tSemantic rank: {res["semantic_rank"]}")
            print("")

        return sorted_results
    elif rerank_method == "batch":
        client = get_llm_client()
        doc_list = ", ".join(
            [f"{idx}. {doc["title"]}" for idx, doc in enumerate(results)]
        )
        logger.debug("DOC LIST => %s", doc_list)
        response = client.models.generate_content(
            model="gemini-3-flash-preview",
            contents=batch_rerank_prompt(query, doc_list),
        )
        assert response.text, "Batch re-ranking LLM response missing"

        try:
            rerank_order = json.loads(response.text)
            rerank_results = [results[i] for i in rerank_order[: int(limit / 5)]]
            for i, res in enumerate(rerank_results):
                print(f"{i+1}. {res["title"]}")
                print(f"\tRerank Rank: {i+1}")
                print(f"\tRRF Score: {res["rrf_score"]:.3f}")
                print(f"\tBM25 rank: {res["bm25_rank"]}")
                print(f"\tSemantic rank: {res["semantic_rank"]}")
                print("")
            return rerank_results
        except Exception as err:
            logger.error(f"Unexpected error parsing json: {err}")
            raise err
    elif rerank_method == "cross_encoder":
        pairs = []
        cross_encoder = CrossEncoder("cross-encoder/ms-marco-TinyBERT-L2-v2")
        for doc in results:
            pairs.append([query, doc["title"]])
        scores = cross_encoder.predict(pairs)

        reranked_results = sorted(
            [
                {**doc, "cross_encoder_score": scores[i]}
                for i, doc in enumerate(results)
            ],
            key=lambda item: item["cross_encoder_score"],
            reverse=True,
        )[: int(limit / 5)]

        for i, res in enumerate(reranked_results):
            print(f"{i+1}. {res["title"]}")
            print(f"\tCross Encoder Score: {res["cross_encoder_score"]:.3f}")
            print(f"\tRRF Score: {res["rrf_score"]:.3f}")
            print(f"\tBM25 rank: {res["bm25_rank"]}")
            print(f"\tSemantic rank: {res["semantic_rank"]}")
            print("")
        return reranked_results
    else:
        raise ValueError("Passed re-rank method value isn't valid")


def get_llm_client() -> genai.Client:
    config = Config()
    assert config.gemini_api_key, "Missing Google Gemini API Key in config"

    client = genai.Client(api_key=config.gemini_api_key)
    return client
