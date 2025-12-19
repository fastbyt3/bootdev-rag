import json
import logging
import math
import os
import pickle
import string
from collections import Counter, defaultdict

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
    """

    def __init__(self) -> None:
        self.index: defaultdict[str, set[int]] = defaultdict(set)
        self.docmap: dict[int, dict] = {}
        self.term_frequencies: dict[int, Counter] = defaultdict(Counter)
        self.cache_dir = "cache"
        self.stemmer = None

    def __setup_stemmer(self) -> None:
        if not self.stemmer:
            self.stemmer = PorterStemmer()

    def __add_document(self, doc_id, text):
        self.__setup_stemmer()

        tokens = tokenize_text(self.stemmer, text)

        self.term_frequencies[doc_id] = Counter(tokens)

        for token in tokens:
            self.index[token].add(doc_id)

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

        with open(f"{self.cache_dir}/index.pkl", "wb") as f:
            pickle.dump(self.index, f)

        with open(f"{self.cache_dir}/term_frequencies.pkl", "wb") as f:
            pickle.dump(self.term_frequencies, f)

        with open(f"{self.cache_dir}/docmap.pkl", "wb") as f:
            pickle.dump(self.docmap, f)

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
