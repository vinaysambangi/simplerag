"""Quick sanity checks: Qdrant connectivity and collection stats."""

import sys
from pathlib import Path

PROJ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJ))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(PROJ / ".env")

from src.core.pipeline import get_pipeline  # noqa: E402


def main():
    pipeline = get_pipeline()
    vs = pipeline.vector_store
    print(f"Backend:            {vs.collection_name}")
    print(f"Collection:         {vs.collection_name}")
    print(f"Vectors in store:   {vs.count()}")


if __name__ == "__main__":
    main()