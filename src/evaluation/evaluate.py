"""LangSmith evaluation harness.

Runs the generated Q&A dataset through the current RAG index and reports
correctness/faithfulness scores to LangSmith.

Usage: python -m src.evaluation.evaluate [--dataset PDF_RAG_Benchmark] [--max-examples 100]
"""

import argparse
import json
import sys
from pathlib import Path

from dotenv import load_dotenv
from langsmith import Client, evaluate
from langsmith.evaluation import EvaluationResult, StringEvaluator

PROJ = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJ))

load_dotenv(PROJ / ".env")

from src.config import QA_DATASET_PATH  # noqa: E402
from src.core.pipeline import get_pipeline  # noqa: E402

DEFAULT_DATASET = "PDF_RAG_Benchmark"

JUDGE_PROMPT = """You are grading a RAG answer.

QUESTION:
{question}

REFERENCE ANSWER (ground truth, from the source documents):
{reference}

PREDICTED ANSWER (from the RAG pipeline):
{prediction}

Rate the predicted answer on {dimension} from 0 to 5.
{criteria}
Respond with ONLY a JSON object: {{"score": <0-5>, "reason": "<one sentence>"}}
"""


class _StringScoreEvaluator(StringEvaluator):
    """Judges an answer with the pipeline LLM, returns score + reason."""

    evaluation_name: str = "qa"
    name: str = "qa"
    dimension: str = "correctness"
    criteria: str = ""

    def __init__(self, llm, name: str, dimension: str, criteria: str):
        super().__init__()
        self.evaluation_name = name
        self.name = name
        self.dimension = dimension
        self.criteria = criteria
        self.llm = llm

    def evaluate_strings(
        self,
        *,
        prediction: str,
        reference: str | None = None,
        input: str | None = None,  # noqa: A002 - langsmith API param name
        **kwargs,
    ) -> EvaluationResult:
        import json as _json
        import re

        prompt = JUDGE_PROMPT.format(
            question=input or "",
            reference=reference or "",
            prediction=prediction,
            dimension=self.dimension,
            criteria=self.criteria,
        )
        raw = self.llm.invoke(prompt).content.strip()
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if not match:
            return EvaluationResult(key=self.evaluation_name, score=0.0, comment=raw[:200])
        try:
            verdict = _json.loads(match.group(0))
        except _json.JSONDecodeError:
            return EvaluationResult(key=self.evaluation_name, score=0.0, comment=raw[:200])
        return EvaluationResult(
            key=self.evaluation_name,
            score=float(verdict.get("score", 0)) / 5.0,
            comment=verdict.get("reason", ""),
        )


def build_dataset(client: Client, dataset_name: str, max_examples: int | None = None) -> str:
    """Create/refresh the LangSmith dataset from the local QA JSONL."""
    if not QA_DATASET_PATH.exists():
        raise SystemExit(f"Dataset file not found: {QA_DATASET_PATH}")

    records = []
    for line in QA_DATASET_PATH.read_text(encoding="utf-8").splitlines():
        try:
            rec = json.loads(line)
            if rec.get("answer"):
                records.append(rec)
        except Exception:
            continue
    if max_examples:
        records = records[:max_examples]
    print(f"Loaded {len(records)} QA records from {QA_DATASET_PATH}")

    if client.has_dataset(dataset_name=dataset_name):
        client.delete_dataset(dataset_name=dataset_name)
    dataset = client.create_dataset(
        dataset_name=dataset_name,
        description="Generated Q&A pairs from the OEM API documentation",
    )
    client.create_examples(
        inputs=[{"input": r["question"]} for r in records],
        outputs=[{"answer": r["answer"]} for r in records],
        dataset_id=dataset.id,
    )
    print(f"Dataset '{dataset_name}' created with {len(records)} examples")
    return dataset_name


def predict_rag(inputs: dict) -> dict:
    query = inputs["input"]
    response = get_pipeline().answer_query(query)
    return {"output": response}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument("--experiment-prefix", default="hybrid-rag-eval")
    args = parser.parse_args()

    client = Client()
    dataset_name = build_dataset(client, args.dataset, args.max_examples)

    llm = get_pipeline().generator.llm
    evaluators = [
        _StringScoreEvaluator(
            llm,
            name="correctness",
            dimension="correctness",
            criteria=(
                "A perfect 5 means the prediction matches the reference answer's "
                "facts exactly. Deduct for wrong facts, missing key details, or "
                "added unsupported information."
            ),
        ),
        _StringScoreEvaluator(
            llm,
            name="faithfulness",
            dimension="faithfulness",
            criteria=(
                "A perfect 5 means the prediction is fully supported by the "
                "reference answer and does not add unsupported facts."
            ),
        ),
    ]

    results = evaluate(
        predict_rag,
        data=dataset_name,
        evaluators=evaluators,
        experiment_prefix=args.experiment_prefix,
        max_concurrency=4,
    )
    print("Evaluation complete. Check the LangSmith UI for metrics.")


if __name__ == "__main__":
    main()