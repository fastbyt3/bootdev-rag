import argparse
import logging
import os
import sys

from inverted_index import InvertedIndex

log_level = os.getenv("LOG_LEVEL", "INFO").upper()
level = getattr(logging, log_level, logging.INFO)

logging.basicConfig(
    level=level,
    format="%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s",
    handlers=[logging.StreamHandler()],
)

logger = logging.getLogger(__name__)


def search_index(query: str):
    inverted_index = InvertedIndex()
    try:
        inverted_index.load()
    except FileNotFoundError:
        print("Required inverted index files not found..")
        sys.exit(1)

    res = set()
    for word in query.strip().lower().split(" "):
        matching_movies = inverted_index.get_documents(word)
        for match in matching_movies:
            res.add(match)

    return [(movie_id, inverted_index.docmap[movie_id]["title"]) for movie_id in res]


def tf_command(doc_id: int, term: str) -> None:
    inverted_index = InvertedIndex()
    inverted_index.load()

    print(f"{doc_id} - {term} -> {inverted_index.get_tf(doc_id, term)}")


def idf_command(term: str):
    """
    calculate inverse doc frequency for a term across dataset
    formula: `math.log((total_doc_count + 1) / (term_match_doc_count + 1))`
    """
    inverted_index = InvertedIndex()
    inverted_index.load()

    idf = inverted_index.get_idf(term)
    print(f"Inverse document frequency of '{term}': {idf:.2f}")


def tfidf_command(doc_id: int, term: str):
    inverted_index = InvertedIndex()
    inverted_index.load()

    tf = inverted_index.get_tf(doc_id, term)
    idf = inverted_index.get_idf(term)

    logger.debug(f"TF-IDF Calculation -> TF = {tf}; IDF = {idf}")
    tf_idf = tf * idf

    print(f"TF-IDF score of '{term}' in document '{doc_id}': {tf_idf:.2f}")


def bm25_idf_command(term: str):
    inverted_index = InvertedIndex()
    inverted_index.load()

    bm25_score = inverted_index.get_bm25_idf(term)
    print(f"BM25 IDF score of '{term}': {bm25_score:.2f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    search_parser.add_argument("query", type=str, help="Search query")

    subparsers.add_parser("build", help="Builds the inverted index of movies data")

    tf_parser = subparsers.add_parser(
        "tf", help="Get term frequency for a specific term in document"
    )
    tf_parser.add_argument("document_id", help="Document to lookup", type=int)
    tf_parser.add_argument("term", help="Term to find freq for in doc", type=str)

    idf_parser = subparsers.add_parser(
        "idf", help="Get inverse document freq for term across dataset"
    )
    idf_parser.add_argument("term", help="Calculate IDF for term")

    tfidf_parser = subparsers.add_parser(
        "tfidf", help="Calculate TF-IDF for a word wrt to doc"
    )
    tfidf_parser.add_argument("document_id", type=int, help="document id")
    tfidf_parser.add_argument("term", help="term to calculate value for")

    bm25_parser = subparsers.add_parser("bm25idf", help="calculate BM25 score for term")
    bm25_parser.add_argument("term")

    args = parser.parse_args()

    match args.command:
        case "build":
            inverted_index = InvertedIndex()
            logger.info("Initializing inverted index")
            inverted_index.build("data/movies.json")
            logger.info("Finished creating inverted index")
            inverted_index.save()
            logger.info("Saved inverted index to disk")
        case "tf":
            tf_command(args.document_id, args.term)
        case "idf":
            idf_command(args.term)
        case "tfidf":
            tfidf_command(args.document_id, args.term)
        case "bm25idf":
            bm25_idf_command(args.term)
        case "search":
            print(f"Searching for: {args.query}")
            matches = search_index(args.query)
            print(matches)
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
