import argparse

from lib.hybrid_search import rrf_search_command
from lib.prompts import (
    answer_questions_prompt,
    augmented_generation,
    summarize_prompt,
    summary_with_citations_prompt,
)


def main():
    parser = argparse.ArgumentParser(description="Retrieval Augmented Generation CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    rag_parser = subparsers.add_parser(
        "rag", help="Perform RAG (search + generate answer)"
    )
    rag_parser.add_argument("query", type=str, help="Search query for RAG")

    summarize_parser = subparsers.add_parser(
        "summarize", help="Summarize results of search"
    )
    summarize_parser.add_argument("query", type=str, help="Search query")
    summarize_parser.add_argument(
        "--limit", type=int, help="Limit number of results (default=5)", default=5
    )

    citations_parser = subparsers.add_parser(
        "citations", help="Provide summary with citations"
    )
    citations_parser.add_argument("query", type=str, help="Search query")
    citations_parser.add_argument(
        "--limit", type=int, help="Limit number of results (default=5)", default=5
    )

    question_parser = subparsers.add_parser(
        "question", help="Provide summary with question"
    )
    question_parser.add_argument("query", type=str, help="Search query")
    question_parser.add_argument(
        "--limit", type=int, help="Limit number of results (default=5)", default=5
    )

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
        case "summarize":
            search_results = rrf_search_command(args.query, limit=args.limit)
            formatted_result = chr(10).join(
                [
                    f"{i}. {res.get("title", "")} - {res.get("document", "")}"
                    for i, res in enumerate(search_results["results"], start=1)
                ]
            )
            llm_summary = summarize_prompt(args.query, formatted_result)

            print("Search Results:")
            for res in search_results["results"]:
                print(f"\t- {res.get("title", "")}")
            print(f"\nLLM Summary:\n{llm_summary}")
        case "citations":
            search_results = rrf_search_command(args.query, limit=args.limit)
            formatted_result = chr(10).join(
                [
                    f"{i}. {res.get("title", "")} - {res.get("document", "")}"
                    for i, res in enumerate(search_results["results"], start=1)
                ]
            )
            llm_summary = summary_with_citations_prompt(args.query, formatted_result)

            print("Search Results:")
            for res in search_results["results"]:
                print(f"\t- {res.get("title", "")}")
            print(f"\nLLM Answer:\n{llm_summary}")
        case "question":
            search_results = rrf_search_command(args.query, limit=args.limit)
            formatted_result = chr(10).join(
                [
                    f"{i}. {res.get("title", "")} - {res.get("document", "")}"
                    for i, res in enumerate(search_results["results"], start=1)
                ]
            )
            llm_answers = answer_questions_prompt(args.query, formatted_result)

            print("Search Results:")
            for res in search_results["results"]:
                print(f"\t- {res.get("title", "")}")
            print(f"\nAnswer:\n{llm_answers}")
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
