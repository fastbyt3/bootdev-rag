import argparse

from lib.hybrid_search import normalize
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

    args = parser.parse_args()

    match args.command:
        case "normalize":
            normalized_scores = normalize(args.score_list)
            for score in normalized_scores:
                print(f"* {score:.4f}")
        case _:
            parser.print_help()


if __name__ == "__main__":
    setup_logging()
    main()
