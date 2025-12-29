import json
import logging
import os
from collections import defaultdict

import numpy as np
from constants import MOVIES_DATA_FILE
from lib.semantic_search import SemanticSearch, cosine_similarity, semantic_chunk

logger = logging.getLogger(__name__)


class ChunkedSemanticSearch(SemanticSearch):
    def __init__(self, model_name="all-MiniLM-L6-v2") -> None:
        super().__init__(model_name)
        self.chunk_embeddings = None
        self.chunk_metadata = None
        self.chunk_size = 4
        self.overlap_size = 1
        self.chunk_embeddings_cache_file = "cache/chunk_embeddings.npy"
        self.chunk_metadata_cache_file = "cache/chunk_metadata.json"

    def build_chunk_embeddings(self, documents):
        self.documents = documents

        all_chunks = []
        chunk_metadata = []

        for doc in documents:
            self.document_map[doc["id"]] = doc

            text = doc.get("description", "")
            if not text.strip():
                logger.debug(f"No description available for movie id = {doc["id"]}")
                continue

            curr_doc_chunks = semantic_chunk(text, self.chunk_size, self.overlap_size)
            curr_doc_chunks = [chunk for chunk in curr_doc_chunks if chunk]

            all_chunks.extend(curr_doc_chunks)

            curr_doc_chunk_count = len(curr_doc_chunks)

            for idx in range(curr_doc_chunk_count):
                chunk_metadata.append(
                    {
                        "movie_idx": doc["id"],
                        "chunk_idx": idx,
                        "total_chunks": curr_doc_chunk_count,
                    }
                )

        # logger.info(f"All chunks: {all_chunks}")
        self.chunk_embeddings = self.model.encode(all_chunks, show_progress_bar=True)

        self.chunk_metadata = chunk_metadata

        np.save(open(self.chunk_embeddings_cache_file, "wb"), self.chunk_embeddings)

        json.dump(
            {"chunks": chunk_metadata, "total_chunks": len(all_chunks)},
            open(self.chunk_metadata_cache_file, "w"),
            indent=2,
        )

        return self.chunk_embeddings

    def load_or_create_chunk_embeddings(self, documents: list[dict]) -> np.ndarray:
        self.documents = documents

        for doc in documents:
            self.document_map[doc["id"]] = doc

        if os.path.exists(self.chunk_embeddings_cache_file) and os.path.exists(
            self.chunk_metadata_cache_file
        ):
            self.chunk_embeddings = np.load(
                open(self.chunk_embeddings_cache_file, "rb")
            )
            assert self.chunk_embeddings is not None

            self.chunk_metadata = json.load(open(self.chunk_metadata_cache_file, "rb"))[
                "chunks"
            ]

            return self.chunk_embeddings
        else:
            return self.build_chunk_embeddings(documents)

    def search_chunks(self, query: str, limit: int = 10):
        query_embedding = self.generate_embedding(query)
        chunk_scores = []

        assert self.chunk_embeddings is not None
        assert self.chunk_metadata is not None

        for idx, chunk_embedding in enumerate(self.chunk_embeddings):
            similarity = cosine_similarity(chunk_embedding, query_embedding)
            chunk_scores.append(
                {
                    "doc_id": self.chunk_metadata[idx]["movie_idx"],
                    "score": similarity,
                    "chunk_metadata": self.chunk_metadata[idx],
                }
            )

        document_scores = defaultdict(float)

        for chunk_score in chunk_scores:
            document_scores[chunk_score["doc_id"]] = max(
                document_scores[chunk_score["doc_id"]], chunk_score["score"]
            )

        sorted_chunk_similarities = sorted(
            document_scores.items(), key=lambda item: item[1], reverse=True
        )[:limit]

        return [
            {
                "id": doc_id,
                "title": self.document_map[doc_id]["title"],
                "description": self.document_map[doc_id]["description"],
                # "document": self.document_map[doc_id],
                "score": round(score, 4),
            }
            for doc_id, score in sorted_chunk_similarities
        ]


def embed_chunks():
    chunked_semantic_search = ChunkedSemanticSearch()
    documents = json.load(open(MOVIES_DATA_FILE))["movies"]
    embeddings = chunked_semantic_search.load_or_create_chunk_embeddings(documents)
    print(f"Generated {len(embeddings)} chunked embeddings")


def search_chunked(query: str, limit: int):
    chunked_semantic_search = ChunkedSemanticSearch()
    documents = json.load(open(MOVIES_DATA_FILE))["movies"]
    chunked_semantic_search.load_or_create_chunk_embeddings(documents)
    results = chunked_semantic_search.search_chunks(query, limit)

    for idx, res in enumerate(results):
        print(f"\n{idx}. {res["title"]} (score: {res["score"]:.4f})")
        # print(f"   {DESCRIPTION}...")
