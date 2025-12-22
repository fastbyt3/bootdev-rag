import argparse

from lib.logging import setup_logging
from lib.semantic_search import (
    chunk_text,
    embed_query_text,
    embed_text,
    search,
    semantic_chunk,
    verify_embeddings,
    verify_model,
)

setup_logging()


def main():
    parser = argparse.ArgumentParser(description="Semantic Search CLI")

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    subparsers.add_parser("verify", help="Print sematic search model info")

    embed_text_parser = subparsers.add_parser(
        "embed_text", help="Generate embedding for a word"
    )
    embed_text_parser.add_argument("text", type=str, help="Text to encode")

    subparsers.add_parser(
        "verify_embeddings", help="valiadte if embeddings are being generated"
    )

    embedquery_parser = subparsers.add_parser(
        "embedquery", help="convert user query to embedding vectors"
    )
    embedquery_parser.add_argument("text", type=str, help="user query")

    search_parser = subparsers.add_parser("search", help="Search similar data")
    search_parser.add_argument("query", type=str, help="query")
    search_parser.add_argument("-l", "--limit", type=int, help="limit results")

    chunk_parser = subparsers.add_parser("chunk", help="Chunk text data")
    chunk_parser.add_argument("text", type=str, help="text to chunk")
    chunk_parser.add_argument("--chunk-size", type=int, help="chunk size", default=200)
    chunk_parser.add_argument("--overlap", type=int, help="chunk overlap", default=0)

    semantic_chunk_parser = subparsers.add_parser(
        "semantic_chunk", help="Chunk text data"
    )
    semantic_chunk_parser.add_argument("text", type=str, help="text to chunk")
    semantic_chunk_parser.add_argument(
        "--max-chunk-size", type=int, help="max chunk size", default=4
    )
    semantic_chunk_parser.add_argument(
        "--overlap", type=int, help="chunk overlap", default=0
    )

    args = parser.parse_args()

    match args.command:
        case "verify":
            verify_model()
        case "embed_text":
            embed_text(args.text)
        case "verify_embeddings":
            verify_embeddings()
        case "embedquery":
            embed_query_text(args.text)
        case "search":
            search(args.query, args.limit)
        case "chunk":
            chunk_text(args.text, args.chunk_size, args.overlap)
        case "semantic_chunk":
            semantic_chunk(args.text, args.max_chunk_size, args.overlap)
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
