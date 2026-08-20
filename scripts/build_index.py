"""Build / rebuild the vector store index from data/pdf and data/text_files.

Usage:
    python scripts/build_index.py              # reset + rebuild
    python scripts/build_index.py --no-reset    # keep existing vectors
"""

import argparse
import sys
from pathlib import Path

PROJ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJ))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(PROJ / ".env")

from src.core.pipeline import get_pipeline  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-reset", action="store_true", help="don't wipe the collection before indexing")
    args = parser.parse_args()

    pipeline = get_pipeline()
    count = pipeline.build_index(reset=not args.no_reset)
    print(f"\nIndex ready: {count} chunks in '{pipeline.vector_store.collection_name}'")


if __name__ == "__main__":
    main()