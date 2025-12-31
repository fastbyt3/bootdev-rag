import argparse
import mimetypes

from lib.prompts import describe_image


def main() -> None:
    parser = argparse.ArgumentParser(description="Image description CLI")
    parser.add_argument(
        "--image",
        type=str,
        help="Image file path",
    )
    parser.add_argument("--query", type=str, help="Query regarding image")

    args = parser.parse_args()

    mime, _ = mimetypes.guess_type(args.image)
    mime = mime or "image/jpeg"

    with open(args.image, "rb") as f:
        contents = f.read()

    result = describe_image(contents, mime, args.query)

    print(f"Rewritten query: {result["response"].strip()}")
    print(f"Total tokens:    {result["total_token_count"]}")


if __name__ == "__main__":
    main()
