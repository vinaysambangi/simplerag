"""End-to-end smoke test: ingest one PDF -> chunk -> embed -> store -> retrieve.

Uses Chroma in a temp dir, local embeddings and semantic chunking so no
network calls are needed. Run:
    .venv\\Scripts\\python.exe scripts\\smoke_test.py
"""

import os
import shutil
import sys
import tempfile
from pathlib import Path

PROJ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJ))

os.environ["VECTOR_DB"] = "CHROMA"
os.environ["EMBEDDING_PROVIDER"] = "local"
os.environ["EMBEDDING_MODEL_NAME"] = "all-MiniLM-L6-v2"
os.environ["CHUNK_STRATEGY"] = "semantic"
os.environ["RERANK_ENABLED"] = "false"
os.environ["RESET_VECTOR_STORE"] = "true"
os.environ["HYBRID_SEARCH"] = "true"

from src import config  # noqa: E402

tmpdir = Path(tempfile.mkdtemp())
config.VECTOR_DIR = tmpdir / "vector_store"
config.PDF_DIR = tmpdir / "pdf"
config.TEXT_DIR = tmpdir / "text_files"
config.PDF_DIR.mkdir(parents=True)
config.TEXT_DIR.mkdir(parents=True)

source_pdf = next(Path(PROJ / "data" / "pdf").glob("*.pdf"))
shutil.copy(source_pdf, config.PDF_DIR / source_pdf.name)
print(f"Using {source_pdf.name}")

from src.core.pipeline import get_pipeline  # noqa: E402

pipeline = get_pipeline()

print("\n=== BUILD INDEX ===")
count = pipeline.build_index(reset=True)
assert count > 0, "no chunks indexed"
print(f"indexed {count} chunks")

print("\n=== RETRIEVE ===")
from src.retrieval import RAGRetriever  # noqa: E402

retriever = pipeline.retriever
docs = retriever.retrieve("how do I send a command to the device", top_k=3)
print(f"retrieved {len(docs)} docs")
for d in docs:
    print(f"  [{d['similarity_score']:.3f}] {d['metadata'].get('source', '?')}")
assert len(docs) > 0, "retrieval returned nothing"

print("\n=== CHAT MESSAGES ===")
from src.generation.prompts import build_chat_messages  # noqa: E402

messages = build_chat_messages("what is the request format for this command?", docs, [])
print(f"system prompt mode: {'api' if '## Request Format' in messages[0]['content'] else 'other'}")
assert messages[0]["role"] == "system"
assert "Context chunks" in messages[-1]["content"]

print("\nSMOKE TEST PASSED")