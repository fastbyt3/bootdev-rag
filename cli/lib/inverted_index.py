import json
import logging
import math
import os
import pickle
import string
from collections import Counter, defaultdict

from constants import BM25_B, BM25_K1
from nltk.stem import PorterStemmer

logger = logging.getLogger(__name__)

# punctuation_table = str.maketrans("", "", "!?.,\\\"'")
stop_words = json.loads(open("data/stop_words.json", "r").read())


def load_movies_data(file_path: str) -> list[dict]:
    with open(file_path, "r") as file:
        return json.load(file)["movies"]


def pre_process_term(term: str) -> str:
    term = term.lower()
    term = term.translate(str.maketrans("", "", string.punctuation))
    return term


def tokenize_text(stemmer: PorterStemmer | None, text: str) -> list[str]:
    assert stemmer

    tokens = []
    for term in text.split():
        processed = pre_process_term(term)
        if not processed or processed in stop_words:
            continue
        stemmed = stemmer.stem(processed)
        if not processed:
            continue
        tokens.append(stemmed)

    return tokens


class InvertedIndex:
    """
    Similar to database index but for text searches
    We map the value to the locations where this value is seen

    Attributes:
        index (dict[str, list[int]]): Map token to document IDs
        docmap (dict[int, Unknown]): Map document ID to respective document entry
        term_frequencies (dict[str, Counter]): Map document ID to term frequency counter
        cache_dir (str): directory to store index snapshots
        doc_lengths (dict): track document length
    """

    def __init__(self) -> None:
        self.index: defaultdict[str, set[int]] = defaultdict(set)
        self.docmap: dict[int, dict] = {}
        self.term_frequencies: dict[int, Counter] = defaultdict(Counter)
        self.cache_dir = "cache"
        self.stemmer = None
        self.doc_lengths: dict[int, int] = {}
        self.index_cache_file = "cache/index.pkl"
        self.term_frequencies_cache_file = "cache/term_frequencies.pkl"
        self.docmap_cache_file = "cache/docmap.pkl"
        self.doclengths_cache_file = "cache/doc_lengths.pkl"

    def __setup_stemmer(self) -> None:
        if not self.stemmer:
            self.stemmer = PorterStemmer()

    def __add_document(self, doc_id, text):
        self.__setup_stemmer()

        tokens = tokenize_text(self.stemmer, text)

        self.term_frequencies[doc_id] = Counter(tokens)

        self.doc_lengths[doc_id] = len(tokens)

        for token in tokens:
            self.index[token].add(doc_id)

    def __get_avg_doc_length(self) -> float:
        doc_count = len(self.doc_lengths)
        if doc_count == 0:
            return 0
        sum = 0
        for doc_length in self.doc_lengths.values():
            sum += doc_length
        return sum / doc_count

    def get_documents(self, term: str) -> list[int]:
        term = term.strip().lower()
        res = list(self.index[term])
        res.sort()
        return res

    def build(self, file_path: str):
        movies_data = load_movies_data(file_path)
        self.stemmer = PorterStemmer()
        for movie in movies_data:
            self.__add_document(movie["id"], f"{movie['title']} {movie['description']}")
            self.docmap[movie["id"]] = movie

    def save(self):
        if not os.path.exists(self.cache_dir):
            os.mkdir(self.cache_dir)

        with open(f"{self.index_cache_file}", "wb") as f:
            pickle.dump(self.index, f)

        with open(f"{self.cache_dir}", "wb") as f:
            pickle.dump(self.term_frequencies, f)

        with open(f"{self.docmap_cache_file}", "wb") as f:
            pickle.dump(self.docmap, f)

        with open(f"{self.doclengths_cache_file}", "wb") as f:
            pickle.dump(self.doc_lengths, f)

    def load(self):
        if not os.path.exists(self.cache_dir):
            raise FileNotFoundError("Cache directory does not exist")
        if not os.path.exists(f"{self.cache_dir}/index.pkl") or not os.path.exists(
            f"{self.cache_dir}/docmap.pkl"
        ):
            raise FileNotFoundError("Index or docmap file does not exist in cache")

        with open(f"{self.cache_dir}/index.pkl", "rb") as f:
            self.index = pickle.load(f)
        with open(f"{self.cache_dir}/docmap.pkl", "rb") as f:
            self.docmap = pickle.load(f)
        with open(f"{self.cache_dir}/term_frequencies.pkl", "rb") as f:
            self.term_frequencies = pickle.load(f)
        with open(f"{self.cache_dir}/doc_lengths.pkl", "rb") as f:
            self.doc_lengths = pickle.load(f)

    def get_tf(self, doc_id: int, term: str) -> int:
        if doc_id not in self.term_frequencies:
            return 0

        self.__setup_stemmer()

        tokens = tokenize_text(self.stemmer, term)
        assert len(tokens) == 1

        token = tokens[0]

        return self.term_frequencies[doc_id][token]

    def get_idf(self, term: str) -> float:
        self.__setup_stemmer()

        tokens = tokenize_text(self.stemmer, term)
        assert len(tokens) == 1
        token = tokens[0]

        total_doc_count = len(self.docmap.keys())

        term_match_doc_count = len(self.index[token])

        idf = math.log((total_doc_count + 1) / (term_match_doc_count + 1))

        return idf

    def get_bm25_idf(self, term: str) -> float:
        self.__setup_stemmer()

        tokens = tokenize_text(self.stemmer, term)
        assert len(tokens) == 1
        token = tokens[0]

        N = len(self.docmap.keys())
        df = len(self.index[token])
        return math.log((N - df + 0.5) / (df + 0.5) + 1)

    def get_bm25_tf(self, doc_id, term, k1, b) -> float:
        tf = self.get_tf(doc_id, term)

        avg_doc_length = self.__get_avg_doc_length()
        length_normalization = 1 - b + b * (self.doc_lengths[doc_id] / avg_doc_length)

        bm25_tf = (tf * (k1 + 1)) / (tf + k1 * length_normalization)
        return bm25_tf

    def bm25(self, doc_id, term) -> float:
        bm25_tf = self.get_bm25_tf(doc_id, term, BM25_K1, BM25_B)
        bm25_idf = self.get_bm25_idf(term)

        return bm25_tf * bm25_idf

    def bm25_search(self, query: str, limit: int) -> list:
        logger.debug(f"QUERY = {query}")

        self.__setup_stemmer()
        query_tokens = tokenize_text(self.stemmer, query)

        scores: defaultdict[int, float] = defaultdict(float)

        for token in query_tokens:
            matching_docs = self.index[token]
            for doc in matching_docs:
                scores[doc] = scores[doc] + self.bm25(doc, token)

        logger.debug(f"Scores = {scores}")

        scores_list = sorted(scores.items(), key=lambda item: item[1], reverse=True)[
            :limit
        ]
        res = [
            {
                "doc_id": doc_id,
                "title": self.docmap[doc_id]["title"],
                "description": self.docmap[doc_id]["description"],
                "score": score,
            }
            for doc_id, score in scores_list
        ]

        return res
