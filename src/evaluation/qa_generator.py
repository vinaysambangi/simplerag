"""Bulk Q&A dataset generation.

Two phases:
  1. QUESTIONS — derive large numbers of developer-style questions from the
     indexed documents (diverse question styles, one JSON array per batch).
  2. ANSWERS — run every question through the RAG pipeline, tracing each run
     in LangSmith, and persist (question, answer, sources) pairs.

Both phases are resume-safe: completed q_idx values are skipped, so the
script can be re-run after an interruption without duplicating work.
"""

import argparse
import hashlib
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_text_splitters import MarkdownHeaderTextSplitter
from langsmith import traceable

PROJ = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJ))

load_dotenv(PROJ / ".env")

from src.config import QA_DATASET_PATH  # noqa: E402
from src.core.pipeline import RagPipeline  # noqa: E402
from src.ingestion import load_documents  # noqa: E402

MIN_SECTION_LEN = 120
SECTION_PREVIEW = 900
SECTIONS_PER_BATCH = 3
QUESTIONS_PER_SECTION_PER_CALL = 3

# Different question styles, cycled across passes to maximize variety.
QUESTION_STYLES = [
    (
        "standard",
        "Write a natural developer question whose answer is fully contained in the section. "
        "Ask as a developer would (how-to, what-is, parameter meaning, configuration). "
        "Use at least one distinctive term/command/endpoint name from the section.",
    ),
    (
        "edge_case",
        "Write a question probing edge cases: exact limits, defaults, allowed values, "
        "error handling, failure modes, timeout behavior, or fallbacks mentioned in the section. "
        "Use at least one distinctive term from the section.",
    ),
    (
        "comparative",
        "Write a question that compares or relates things within the section "
        "(e.g. differences between two commands/parameters/versions/encodings mentioned there). "
        "Use at least one distinctive term from the section.",
    ),
    (
        "troubleshooting",
        "Write a question phrased as a real integration problem a driver developer would hit "
        "(e.g. 'why is X not working', 'how do I fix Y') whose answer is directly contained in "
        "the section. Use at least one distinctive term from the section.",
    ),
]


class QADatasetGenerator:
    def __init__(
        self,
        questions_path: Path = PROJ / "data" / "qa" / "questions.jsonl",
        dataset_path: Path = QA_DATASET_PATH,
        gen_model: str = "gpt-4o-mini",
    ):
        self.questions_path = questions_path
        self.dataset_path = dataset_path
        self.questions_path.parent.mkdir(parents=True, exist_ok=True)
        self.dataset_path.parent.mkdir(parents=True, exist_ok=True)
        self.gen_llm = ChatOpenAI(
            model=gen_model,
            temperature=0.7,
            api_key=os.getenv("OPENAI_API_KEY"),
        )
        self.pipeline = RagPipeline()

    # ------------------------------------------------------------------
    # Phase 1: questions
    # ------------------------------------------------------------------
    def collect_sections(self) -> List[Dict]:
        docs = load_documents()
        splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=[
                ("#", "endpoint"),
                ("##", "block_type"),
                ("###", "field_group"),
            ],
            strip_headers=False,
        )
        sections, seen = [], set()
        for d in docs:
            for s in splitter.split_text(d.page_content):
                text = s.page_content.strip()
                if len(text) < MIN_SECTION_LEN:
                    continue
                key = hashlib.md5(text.encode("utf-8", "ignore")).hexdigest()
                if key in seen:
                    continue
                seen.add(key)
                sections.append({"text": text, "metadata": dict(s.metadata)})
        sections.sort(key=lambda x: len(x["text"]), reverse=True)
        return sections

    def _load_done(self, path: Path) -> Dict[int, Dict]:
        done = {}
        if path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
                try:
                    rec = json.loads(line)
                    done[rec["q_idx"]] = rec
                except Exception:
                    continue
        return done

    def _format_section(self, idx, section) -> str:
        meta = section["metadata"]
        trail = " / ".join(
            str(meta[k]) for k in ("endpoint", "block_type", "field_group") if k in meta
        )
        header = f"[{idx}] (section: {trail or 'body'})" if trail else f"[{idx}]"
        return f"{header}\n{section['text'][:SECTION_PREVIEW]}"

    def _parse_json_array(self, response) -> list:
        match = re.search(r"\[.*?\]", response, re.DOTALL)
        if not match:
            return []
        try:
            return json.loads(match.group(0))
        except Exception:
            return []

    def generate_questions(
        self, target: int, max_passes: int = 12, stop_after: int = 600
    ) -> int:
        """Generate up to `target` unique questions. Returns count written."""
        sections = self.collect_sections()
        print(f"Collected {len(sections)} unique sections")
        if not sections:
            print("No sections found — check data/pdf and data/text_files.")
            return 0

        # Weight sections by length so longer ones yield more questions.
        weights = [max(1, min(12, len(s["text"]) // 450)) for s in sections]
        capacity = sum(weights)
        print(f"Estimated question capacity: ~{capacity}")

        done = self._load_done(self.questions_path)
        total = len(done)

        for style_name, style_rule in QUESTION_STYLES:
            if total >= target:
                break
            system = (
                "You are a developer integrating these devices via their API/protocol documentation. "
                f"Task: {style_rule}\n"
                "Return ONLY a valid JSON array where each element is an object "
                '{"question": "...", "section": N} — one question per numbered section, in order. '
                "Each question must be answerable from that section alone. No extra text."
            )

            for batch_start in range(0, len(sections), SECTIONS_PER_BATCH):
                if total >= target:
                    break
                batch = sections[batch_start : batch_start + SECTIONS_PER_BATCH]
                indices = [
                    i for i in range(batch_start, batch_start + len(batch))
                ]
                sections_text = "\n\n".join(
                    self._format_section(i, s) for i, s in zip(indices, batch)
                )
                try:
                    response = self.gen_llm.invoke(
                        [
                            {"role": "system", "content": system},
                            {"role": "user", "content": "Sections:\n" + sections_text},
                        ]
                    ).content
                except Exception as e:
                    print(f"  Gen call failed: {e}; skipping batch")
                    time.sleep(1)
                    continue

                parsed = self._parse_json_array(response)
                new_count = 0
                with self.questions_path.open("a", encoding="utf-8") as f:
                    for item in parsed:
                        if total >= target:
                            break
                        if new_count >= len(indices):
                            # LLM returned more questions than sections in
                            # this batch — ignore the extras.
                            break
                        if isinstance(item, str):
                            question, sec_idx = item, indices[new_count]
                        elif isinstance(item, dict):
                            question = str(item.get("question", "")).strip()
                            raw_sec = item.get("section")
                            try:
                                sec_idx = int(raw_sec) if raw_sec is not None else indices[new_count]
                            except (TypeError, ValueError):
                                sec_idx = indices[new_count]
                        else:
                            continue
                        if len(question) < 10:
                            continue
                        if not (0 <= sec_idx < len(sections)):
                            sec_idx = indices[new_count]
                        q_idx = total
                        record = {
                            "q_idx": q_idx,
                            "question": question,
                            "section": sec_idx,
                            "style": style_name,
                            "source": sections[sec_idx]["metadata"].get(
                                "source", ""
                            ),
                        }
                        f.write(json.dumps(record, ensure_ascii=False) + "\n")
                        f.flush()
                        total += 1
                        new_count += 1
                print(
                    f"  [{style_name}] pass -> {total}/{target} questions", end="\r"
                )
            print()

            if total - len(done) >= stop_after:
                done = self._load_done(self.questions_path)

        print(f"\nQuestions phase complete: {total} total in {self.questions_path}")
        return total

    # ------------------------------------------------------------------
    # Phase 2: answers (traced RAG runs)
    # ------------------------------------------------------------------
    @traceable(name="QA_Answer_Run", run_type="chain", metadata={"phase": "qa_dataset"})
    def _answer_one(self, q_idx: int, question: str) -> Dict:
        answer = self.pipeline.answer_query(question)
        return {"q_idx": q_idx, "question": question, "answer": answer}

    def generate_answers(self, workers: int = 4, max_retries: int = 2) -> int:
        questions = self._load_done(self.questions_path)
        if not questions:
            print("No questions found — run the questions phase first.")
            return 0

        done = self._load_done(self.dataset_path)
        pending = [q for q in questions.values() if q["q_idx"] not in done]
        print(f"Questions: {len(questions)} | already answered: {len(done)} | pending: {len(pending)}")

        if not pending:
            print("Everything already answered.")
            return len(done)

        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(self._answer_one, q["q_idx"], q["question"]): q
                for q in pending
            }
            completed = 0
            for future in as_completed(futures):
                q = futures[future]
                try:
                    result = future.result(timeout=180)
                    with self.dataset_path.open("a", encoding="utf-8") as f:
                        f.write(json.dumps(result, ensure_ascii=False) + "\n")
                    completed += 1
                except Exception as e:
                    print(f"  ERROR q_idx {q['q_idx']}: {e}")
                if completed % 25 == 0:
                    print(f"  {completed}/{len(pending)} answered")

        total = len(self._load_done(self.dataset_path))
        print(f"\nAnswers phase complete: {total} records in {self.dataset_path}")
        return total


def main():
    parser = argparse.ArgumentParser(description="Generate a large Q&A dataset for RAG training/evaluation")
    parser.add_argument("--num", type=int, default=5000, help="target number of questions (default 5000)")
    parser.add_argument("--phase", choices=["questions", "answers", "all"], default="all")
    parser.add_argument("--workers", type=int, default=4, help="parallel answer workers")
    args = parser.parse_args()

    gen = QADatasetGenerator()
    if args.phase in ("questions", "all"):
        gen.generate_questions(args.num)
    if args.phase in ("answers", "all"):
        gen.generate_answers(workers=args.workers)


if __name__ == "__main__":
    main()