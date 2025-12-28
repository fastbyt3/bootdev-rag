import argparse

from lib.hybrid_search import normalize, rrf_search, weighted_search
from lib.logging import setup_logging


def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid Search CLI")
    subparser = parser.add_subparsers(dest="command", help="Available commands")

    normalize_parser = subparser.add_parser(
        "normalize", help="Normalize a list of scores"
    )
    normalize_parser.add_argument(
        "score_list",
        nargs="+",
        type=float,
        help="Space separated (float) scores",
    )

    weighted_search_parser = subparser.add_parser(
        "weighted-search", help="Perform weighted search"
    )
    weighted_search_parser.add_argument("query", type=str, help="Search query")
    weighted_search_parser.add_argument(
        "-l", "--limit", type=int, help="Number of results", default=5
    )
    weighted_search_parser.add_argument(
        "-a",
        "--alpha",
        type=float,
        help="Control weighting between BM25 and Semantic score",
        default=0.5,
    )

    rrf_search_parser = subparser.add_parser("rrf-search", help="Perform rrf search")
    rrf_search_parser.add_argument("query", type=str, help="Search query")
    rrf_search_parser.add_argument(
        "-l", "--limit", type=int, help="Number of results", default=5
    )
    rrf_search_parser.add_argument(
        "-k",
        type=float,
        help="Control weight between higher vs lower ranked results",
        default=60,
    )

    args = parser.parse_args()

    match args.command:
        case "normalize":
            normalized_scores = normalize(args.score_list)
            for score in normalized_scores:
                print(f"* {score:.4f}")
        case "weighted-search":
            weighted_search(args.query, args.alpha, args.limit)
        case "rrf-search":
            rrf_search(args.query, args.k, args.limit)
        case _:
            parser.print_help()


if __name__ == "__main__":
    setup_logging()
    main()
