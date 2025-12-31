#!/usr/bin/env python3

import argparse

from lib.logging import setup_logging
from lib.multimodal_search import image_search, verify_image_embedding


def main() -> None:
    parser = argparse.ArgumentParser(description="Semantic Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    verify_img_embedding_parser = subparsers.add_parser(
        "verify_image_embedding", help="Verify image embedding"
    )
    verify_img_embedding_parser.add_argument(
        "image_path", type=str, help="Path to image file"
    )

    image_search_parser = subparsers.add_parser(
        "image_search", help="Search with image"
    )
    image_search_parser.add_argument("image_path", type=str, help="Path to image file")

    args = parser.parse_args()

    match args.command:
        case "verify_image_embedding":
            embedding = verify_image_embedding(args.image_path)
            print(f"Embedding shape: {embedding.shape[0]} dimensions")
        case "image_search":
            results = image_search(args.image_path)
            print("\n====== IMAGE SEARCH RESULTS =======\n")
            for i, res in enumerate(results, start=1):
                print(
                    f"{i}. {res.get("document").get("title")} (similarity: {res.get("similarity"):.3f})"
                )
                # print(f"\t{res.get("document").get("description")}")
                print("")

        case _:
            parser.print_help()


if __name__ == "__main__":
    setup_logging()
    main()
