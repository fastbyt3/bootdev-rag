import argparse

from lib.hybrid_search import rrf_search_command
from lib.prompts import augmented_generation


def main():
    parser = argparse.ArgumentParser(description="Retrieval Augmented Generation CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    rag_parser = subparsers.add_parser(
        "rag", help="Perform RAG (search + generate answer)"
    )
    rag_parser.add_argument("query", type=str, help="Search query for RAG")

    args = parser.parse_args()

    match args.command:
        case "rag":
            query = args.query
            search_results = rrf_search_command(query, limit=5)
            results_formatted = chr(10).join(
                [
                    f"{i}. {res.get("title", "")} - {res.get("document", "")}"
                    for i, res in enumerate(search_results["results"], start=1)
                ]
            )

            llm_response = augmented_generation(query, results_formatted)

            print("Search Results:")
            for res in search_results["results"]:
                print(f"\t- {res.get("title", "")}")
            print(f"\nRAG Response:\n{llm_response}")
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
