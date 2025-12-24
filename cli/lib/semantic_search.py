import json
import logging
import os
import re
import string

import numpy as np
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot_product / (norm1 * norm2)


class SemanticSearch:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        self.model = SentenceTransformer(model_name)
        self.embeddings = None
        self.documents = None
        self.document_map = {}
        self.embeddings_file = "cache/movie_embeddings.npy"

    def generate_embedding(self, text: str):
        if not text.strip():
            raise ValueError("text seems to be empty")

        # Arg to encode needs to be a list
        embedding = self.model.encode([text])
        return embedding[0]

    def build_embeddings(self, documents):
        self.documents = documents

        tmp_movies_text = []
        for doc in documents:
            self.document_map[doc["id"]] = doc
            tmp_movies_text.append(f"{doc['title']}: {doc['description']}")

        self.embeddings = self.model.encode(tmp_movies_text, show_progress_bar=True)

        np.save(open(self.embeddings_file, "wb"), self.embeddings)

        return self.embeddings

    def load_or_create_embeddings(self, documents):
        self.documents = documents
        for doc in documents:
            self.document_map[doc["id"]] = doc

        if not os.path.exists(self.embeddings_file):
            logger.info("Embeddings cache does not exist, creating new embeddings")
            return self.build_embeddings(documents)

        logger.info("Loading embeddings from cached file")
        self.embeddings = np.load(open(self.embeddings_file, "rb"))

        assert len(self.embeddings) == len(documents)

        return self.embeddings

    def search(self, query, limit):
        assert (
            self.embeddings is not None and len(self.embeddings) > 0
        ), "Embeddings aren't loaded"

        assert self.documents and self.document_map, "Documents aren't loaded"

        query_embedding = self.generate_embedding(query)
        logger.info(f"Generated embeddings for query: {query}")

        query_doc_similarity = sorted(
            [
                (
                    self.documents[idx]["id"],
                    cosine_similarity(doc_embedding, query_embedding),
                )
                for idx, doc_embedding in enumerate(self.embeddings)
            ],
            key=lambda item: item[1],
            reverse=True,
        )[:limit]

        return [
            {
                "score": score,
                "title": self.document_map[id]["title"],
                "description": self.document_map[id]["description"],
            }
            for (id, score) in query_doc_similarity
        ]


def verify_model():
    semantic_search = SemanticSearch()
    print(f"Model loaded: {semantic_search.model}")
    print(f"Max sequence length: {semantic_search.model.max_seq_length}")


def embed_text(text):
    semantic_search = SemanticSearch()
    embedding = semantic_search.generate_embedding(text)
    print(f"Text: {text}")
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Dimensions: {embedding.shape[0]}")


def verify_embeddings():
    semantic_search = SemanticSearch()
    documents = json.load(open("data/movies.json", "r"))["movies"]
    embeddings = semantic_search.load_or_create_embeddings(documents)
    print(f"Number of docs:   {len(documents)}")
    print(
        f"Embeddings shape: {embeddings.shape[0]} vectors in {embeddings.shape[1]} dimensions"
    )


def embed_query_text(query):
    semantic_search = SemanticSearch()
    embedding = semantic_search.generate_embedding(query)
    print(f"Query: {query}")
    print(f"First 5 dimensions: {embedding[:5]}")
    print(f"Shape: {embedding.shape}")


def search(query, limit):
    semantic_search = SemanticSearch()
    documents = json.load(open("data/movies.json", "r"))["movies"]
    semantic_search.load_or_create_embeddings(documents)
    results = semantic_search.search(query, limit)
    for idx, res in enumerate(results):
        print(
            f"{idx + 1}. {res['title']} ({res['score']:.4f})\n\t{res['description']}\n"
        )


def chunk_text(text: str, chunk_size: int, overlap_size: int):
    assert chunk_size > overlap_size, "Chunk size should be greater than overlap size"

    text_splits = text.split()
    chunks = [
        text_splits[i : i + chunk_size]
        for i in range(0, len(text_splits), chunk_size - overlap_size)
    ]
    print(f"Chunking {len(text)} characters")
    for idx, chunk in enumerate(chunks):
        print(f"{idx + 1}. {' '.join(chunk)}")


# def semantic_chunk(text: str, chunk_size: int, overlap_size: int, logging: bool = True):
#     text_splits = re.split(r"(?<=[.!?])\s+", text)
#     chunks = [
#         " ".join(text_splits[i : i + chunk_size])
#         for i in range(0, len(text_splits), chunk_size - overlap_size)
#     ]
#
#     if logging:
#         print(f"Semantically chunking {len(text)} characters")
#         for chunk in chunks:
#             print(chunk)
#
#     return chunks
#


def semantic_chunk(
    text: str,
    max_chunk_size: int,
    overlap: int,
) -> list[str]:
    processed_text = text.strip()
    if not text:
        return []

    sentences = re.split(r"(?<=[.!?])\s+", processed_text)

    sentences = [sentence for sentence in sentences if sentence.strip()]

    n_sentences = len(sentences)

    if n_sentences == 1 and sentences[0][-1] in string.punctuation:
        return [processed_text]

    chunks = []
    i = 0

    while i < n_sentences:
        chunk_sentences = sentences[i : i + max_chunk_size]
        if chunks and len(chunk_sentences) <= overlap:
            break
        chunks.append(" ".join(chunk_sentences))
        i += max_chunk_size - overlap

    return chunks
