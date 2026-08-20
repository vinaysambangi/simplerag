"""Smoke tests for the core pipeline.

Run:  python -m tests.test_pipeline  (no pytest dependency needed)
"""

import sys
import tempfile
import unittest
from pathlib import Path

PROJ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJ))

from src.core.sessions import ChatSessionManager  # noqa: E402
from src.generation.prompts import (  # noqa: E402
    build_chat_messages,
    detect_query_type,
    format_context,
)
from src.chunking.agentic import _segment_units  # noqa: E402


class TestSessionIsolation(unittest.TestCase):
    """A new chat session must never see another session's messages."""

    def setUp(self):
        tmpdir = tempfile.mkdtemp()
        self.manager = ChatSessionManager(Path(tmpdir) / "test.db")

    def test_new_session_has_no_history(self):
        s1 = self.manager.create_session()
        self.assertEqual(self.manager.history_for_chat(s1["id"]), [])

    def test_sessions_are_isolated(self):
        s1 = self.manager.create_session()
        s2 = self.manager.create_session()

        self.manager.add_message(s1["id"], "user", "What is onConnect?")
        self.manager.add_message(s1["id"], "assistant", "It is a handshake.")

        self.assertEqual(len(self.manager.history_for_chat(s1["id"])), 2)
        self.assertEqual(self.manager.history_for_chat(s2["id"]), [])

    def test_history_only_within_session(self):
        s1 = self.manager.create_session()
        self.manager.add_message(s1["id"], "user", "First question")
        self.manager.add_message(s1["id"], "assistant", "First answer")

        s2 = self.manager.create_session()
        self.manager.add_message(s2["id"], "user", "Another question")
        self.manager.add_message(s2["id"], "assistant", "Another answer")

        # s1's history must not include s2's messages
        contents = [m["content"] for m in self.manager.history_for_chat(s1["id"])]
        self.assertEqual(contents, ["First question", "First answer"])


class TestPromptBuilding(unittest.TestCase):
    def test_format_context_includes_rank_and_source(self):
        docs = [
            {
                "rank": 1,
                "content": "hello",
                "metadata": {"source": "doc.pdf"},
                "similarity_score": 0.9,
            }
        ]
        ctx = format_context(docs)
        self.assertIn("[Chunk 1]", ctx)
        self.assertIn("doc.pdf", ctx)
        self.assertIn("hello", ctx)

    def test_build_chat_messages_contains_history_and_query(self):
        docs = [
            {
                "rank": 1,
                "content": "context text",
                "metadata": {"source": "doc.pdf"},
                "similarity_score": 0.9,
            }
        ]
        history = [{"role": "user", "content": "prev q"}, {"role": "assistant", "content": "prev a"}]
        messages = build_chat_messages("new q", docs, history)

        roles = [m["role"] for m in messages]
        self.assertEqual(roles, ["system", "user", "assistant", "user"])
        self.assertEqual(messages[0]["role"], "system")
        self.assertIn("new q", messages[-1]["content"])
        self.assertIn("context text", messages[-1]["content"])


class TestQueryTypeDetection(unittest.TestCase):
    def test_api_question_detected(self):
        self.assertEqual(
            detect_query_type("How do I call the POST /api/v1/controls endpoint?"),
            "api",
        )

    def test_feature_question_detected(self):
        self.assertEqual(
            detect_query_type("What is the sleep mode feature and how does it work?"),
            "feature",
        )

    def test_general_question_detected(self):
        self.assertEqual(
            detect_query_type("Does the document mention anything about constraints?"),
            "general",
        )

    def test_system_prompt_matches_mode(self):
        from src.generation.prompts import system_prompt_for

        api_prompt = system_prompt_for("Show me the request body for POST /api/control")
        self.assertIn("## Request Format", api_prompt)
        feature_prompt = system_prompt_for("Explain the diagnostic feature")
        self.assertIn("## What it is", feature_prompt)


class TestChunkUnitSegmentation(unittest.TestCase):
    def test_code_blocks_are_kept_intact(self):
        text = (
            "Some prose about the endpoint.\n\n"
            "```json\n{\"key\": \"value\"}\n```\n\n"
            "More prose after the example."
        )
        units = _segment_units(text)
        kinds = [k for k, _ in units]
        self.assertIn("code", kinds)
        code_unit = next(u for u in units if u[0] == "code")
        self.assertIn("```json", code_unit[1])
        self.assertIn("{\"key\": \"value\"}", code_unit[1])

    def test_table_lines_stay_together(self):
        text = (
            "| Param | Type |\n"
            "|-------|------|\n"
            "| foo   | int  |\n"
            "prose after"
        )
        units = _segment_units(text)
        kinds = [k for k, _ in units]
        self.assertIn("table", kinds)
        table_unit = next(u for u in units if u[0] == "table")
        self.assertIn("foo", table_unit[1])
        self.assertIn("| Param | Type |", table_unit[1])


class TestBM25Index(unittest.TestCase):
    def test_bm25_ranks_keyword_match_first(self):
        from src.retrieval.bm25 import BM25Index

        index = BM25Index()
        index.refresh(
            [
                {"id": "1", "content": "POST /api/v1/controls sets the motor speed", "metadata": {}},
                {"id": "2", "content": "The device supports power saving modes", "metadata": {}},
                {"id": "3", "content": "Controls are updated asynchronously", "metadata": {}},
                {"id": "4", "content": "Bluetooth pairing and connection states", "metadata": {}},
                {"id": "5", "content": "Firmware version and update procedure", "metadata": {}},
            ]
        )
        hits = index.search("controls endpoint speed", top_k=5)
        self.assertTrue(hits)
        self.assertEqual(hits[0]["doc"]["id"], "1")

    def test_rrf_fusion_merges_rankings(self):
        from src.retrieval.bm25 import rrf_fuse

        fused = rrf_fuse(["a", "b", "c"], ["b", "c", "d"], k=60)
        self.assertEqual(fused[0], "b")
        self.assertIn("a", fused)


class TestMessageAnswerType(unittest.TestCase):
    def test_answer_type_round_trip(self):
        tmpdir = tempfile.mkdtemp()
        manager = ChatSessionManager(Path(tmpdir) / "test.db")
        s = manager.create_session()
        manager.add_message(s["id"], "user", "question")
        manager.add_message(
            s["id"], "assistant", "answer", sources=[{"id": "1"}], answer_type="api"
        )
        messages = manager.list_messages(s["id"])
        self.assertEqual(messages[1]["answer_type"], "api")
        self.assertEqual(messages[1]["sources"], [{"id": "1"}])

    def test_legacy_plain_array_sources_still_load(self):
        import json
        import sqlite3

        tmpdir = tempfile.mkdtemp()
        db_path = Path(tmpdir) / "test.db"
        manager = ChatSessionManager(db_path)
        s = manager.create_session()
        with sqlite3.connect(db_path) as conn:
            conn.execute(
                "INSERT INTO messages (id, session_id, role, content, sources, created_at)"
                " VALUES (?, ?, ?, ?, ?, ?)",
                ("legacy", s["id"], "assistant", "old", json.dumps([{"id": "x"}]), "2024-01-01T00:00:00"),
            )
        messages = manager.list_messages(s["id"])
        self.assertEqual(messages[0]["sources"], [{"id": "x"}])
        self.assertIsNone(messages[0]["answer_type"])


if __name__ == "__main__":
    unittest.main(verbosity=2)