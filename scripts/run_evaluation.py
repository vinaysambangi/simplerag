"""Run LangSmith evaluation on the current index.

Usage:
    python scripts/run_evaluation.py --max-examples 200
"""

import argparse
import sys
from pathlib import Path

PROJ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJ))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(PROJ / ".env")

from src.evaluation.evaluate import main as evaluate_main  # noqa: E402

if __name__ == "__main__":
    evaluate_main()