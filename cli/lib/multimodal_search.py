import logging
import os
import sys

import numpy as np
from lib.search_utils import MULTIMODEL_TEXT_EMEBEDDINGS_PATH, load_movies
from lib.semantic_search import cosine_similarity
from PIL import Image
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


class MultimodalSearch:
    def __init__(self, documents: list[dict], model_name="clip-ViT-B-32"):
        self.model = SentenceTransformer(model_name)
        self.documents = documents
        texts = [f"{doc["title"]}: {doc["description"]}" for doc in documents]
        self.texts = texts
        self.text_embeddings = self._generate_text_embeddings(texts)

    def _generate_text_embeddings(self, documents: list[str]):
        if os.path.exists(MULTIMODEL_TEXT_EMEBEDDINGS_PATH):
            logger.info(
                f"Loading multi model text embeddings from cache. File = {MULTIMODEL_TEXT_EMEBEDDINGS_PATH}"
            )
            return np.load(open(MULTIMODEL_TEXT_EMEBEDDINGS_PATH, "rb"))

        embeddings = self.model.encode(documents, show_progress_bar=True)

        os.makedirs(os.path.dirname(MULTIMODEL_TEXT_EMEBEDDINGS_PATH), exist_ok=True)
        np.save(MULTIMODEL_TEXT_EMEBEDDINGS_PATH, embeddings)
        logger.info("Successfully written multi model text embeddings to cache file")

        return embeddings

    def embed_image(self, image_path: str):
        image = Image.open(image_path)
        embeddings = self.model.encode([image], show_progress_bar=True)  # type: ignore
        return embeddings[0]

    def search_with_image(self, image_path: str):
        image = Image.open(image_path)
        image_embedding = self.model.encode([image], show_progress_bar=True)[0]  # type: ignore

        similarities = []
        for i, text_embedding in enumerate(self.text_embeddings):
            similarity = cosine_similarity(text_embedding, image_embedding)
            similarities.append(
                {"document": self.documents[i], "similarity": similarity}
            )

        return sorted(similarities, key=lambda item: item["similarity"], reverse=True)[
            :5
        ]


def verify_image_embedding(image_path: str):
    multimodel = MultimodalSearch(load_movies())
    embedding = multimodel.embed_image(image_path)
    return embedding


def image_search(image_path: str):
    multimodel = MultimodalSearch(load_movies())
    return multimodel.search_with_image(image_path)
