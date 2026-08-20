"""Generate a large Q&A dataset for RAG training/evaluation.

Phases (both resume-safe):
  1. questions — generate developer-style questions from the documents
  2. answers   — run each question through RAG (traced in LangSmith) and
                 store question/answer/source pairs.

Usage:
    python scripts/generate_qa_dataset.py --num 5000
    python scripts/generate_qa_dataset.py --phase questions --num 5000
    python scripts/generate_qa_dataset.py --phase answers --workers 4
"""

import argparse
import sys
from pathlib import Path

PROJ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJ))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(PROJ / ".env")

from src.evaluation.qa_generator import QADatasetGenerator  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num", type=int, default=5000, help="target question count (default 5000)")
    parser.add_argument("--phase", choices=["questions", "answers", "all"], default="all")
    parser.add_argument("--workers", type=int, default=4, help="parallel workers for the answers phase")
    args = parser.parse_args()

    gen = QADatasetGenerator()
    if args.phase in ("questions", "all"):
        gen.generate_questions(args.num)
    if args.phase in ("answers", "all"):
        gen.generate_answers(workers=args.workers)


if __name__ == "__main__":
    main()