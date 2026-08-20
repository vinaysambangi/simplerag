"""Agentic LLM-driven chunking with structure preservation.

Pipeline per document:
  1. Split on markdown-style headings (endpoint / Request / Response / ...)
     with a MarkdownHeaderTextSplitter — the PDF ingestion step already
     reconstructs headings from font size / numbered sections.
  2. Within each section, segment into atomic units:
       - fenced code blocks (JSON request/response examples) — NEVER split
       - markdown tables — kept as one unit
       - prose runs
  3. Pack units into chunks up to CHUNK_MAX_SIZE, preferring to break
     between units (never inside a code block or table).
  4. Long prose runs are split on topic boundaries:
       - strategy "hybrid" (default): LLM returns sentence indices where a
         new topic starts; on API failure it falls back to semantic
         boundaries (embedding cosine similarity drops).
       - strategy "semantic": pure embedding-similarity boundaries, no LLM.
       - strategy "size": plain size-based splitting.
  5. Every chunk gets a context prefix (section heading path + a short
     description of what the chunk contains) so each chunk is
     self-contained for retrieval.

Metadata on each chunk: source, section_path, kind (prose/code/table/mixed),
chunk_index, content_length.
"""

import json
import os
import re
from typing import List, Tuple

import numpy as np
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_text_splitters import MarkdownHeaderTextSplitter

from ..config import (
    CHUNK_BATCH_SIZE,
    CHUNK_CONTEXT_PREFIX,
    CHUNK_MAX_SIZE,
    CHUNK_MIN_LENGTH,
    CHUNK_STRATEGY,
)

MARKDOWN_HEADERS_TO_SPLIT_ON = [
    ("#", "endpoint"),
    ("##", "block_type"),
    ("###", "field_group"),
]

BATCH_SPLIT_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an expert document chunking agent.\n"
            "Analyze the provided numbered list of consecutive sentences from a document section.\n"
            "Identify sentence indices where a NEW distinct topic or sub-concept begins.\n"
            "Return ONLY a valid JSON array of numbers indicating the 0-based sentence indices that start a new topic.\n"
            "Example output: [4, 12, 19]\n"
            "If the whole text is a single coherent topic, return []. Do not include 0 in the list.",
        ),
        ("user", "Sentences:\n{sentences_text}"),
    ]
)

FENCED_CODE_RE = re.compile(
    r"(```[\w-]*\n.*?```|~~~[\w-]*\n.*?~~~)", re.DOTALL
)
TABLE_LINE_RE = re.compile(r"^\s*\|.*\|\s*$")

# ---------------------------------------------------------------------------
# Unit segmentation
# ---------------------------------------------------------------------------


def _segment_units(text: str) -> List[Tuple[str, str]]:
    """Split a section into atomic units: (kind, text).

    kinds: "code" (fenced block, never split), "table" (consecutive markdown
    table lines, kept together), "prose" (everything else).
    """
    units: List[Tuple[str, str]] = []
    cursor = 0

    for match in FENCED_CODE_RE.finditer(text):
        if match.start() > cursor:
            units.extend(_segment_prose(text[cursor : match.start()]))
        units.append(("code", match.group(0).strip()))
        cursor = match.end()

    if cursor < len(text):
        units.extend(_segment_prose(text[cursor:]))
    return units


def _segment_prose(text: str) -> List[Tuple[str, str]]:
    """Split prose into runs, keeping consecutive table lines together."""
    lines = text.splitlines()
    units: List[Tuple[str, str]] = []
    current_table: List[str] = []
    current_prose: List[str] = []

    def flush_table():
        if current_table:
            units.append(("table", "\n".join(current_table).strip()))
            current_table.clear()

    def flush_prose():
        joined = "\n".join(current_prose).strip()
        if joined:
            units.append(("prose", joined))
        current_prose.clear()

    for line in lines:
        if TABLE_LINE_RE.match(line):
            flush_prose()
            current_table.append(line)
        else:
            flush_table()
            current_prose.append(line)
    flush_table()
    flush_prose()
    return units


# ---------------------------------------------------------------------------
# Topic-boundary splitting for long prose runs
# ---------------------------------------------------------------------------


class TopicSplitter:
    """Splits a long prose unit into smaller pieces on topic boundaries."""

    def __init__(
        self,
        strategy: str = CHUNK_STRATEGY,
        model_name: str = "gpt-4o-mini",
        max_chunk_size: int = CHUNK_MAX_SIZE,
        batch_size: int = CHUNK_BATCH_SIZE,
    ):
        self.strategy = strategy
        self.max_chunk_size = max_chunk_size
        self.batch_size = batch_size
        self._llm = None
        if strategy in ("hybrid", "agentic"):
            self._llm = ChatOpenAI(
                model=model_name,
                temperature=0,
                api_key=os.getenv("OPENAI_API_KEY"),
            )

    def _split_into_sentences(self, text: str) -> List[str]:
        sentences = re.split(r"(?<=[.!?])\s+|\n+", text)
        return [s.strip() for s in sentences if s.strip()]

    def _llm_boundaries(self, sentences: List[str]) -> set:
        """Ask the LLM which sentence indices start a new topic."""
        if self._llm is None:
            return set()
        indices: set = set()
        for i in range(0, len(sentences), self.batch_size):
            batch = sentences[i : i + self.batch_size]
            formatted_input = "\n".join(
                f"[{idx}] {s}" for idx, s in enumerate(batch)
            )
            try:
                response = self._llm.invoke(
                    {"sentences_text": formatted_input}
                ).content.strip()
                match = re.search(r"\[.*?\]", response, re.DOTALL)
                if match:
                    indices.update(json.loads(match.group(0)))
            except Exception as e:  # noqa: BLE001
                print(f"Warning: LLM boundary detection failed ({e})")
                return set()
        return indices

    def _semantic_boundaries(
        self,
        sentences: List[str],
        embedding_manager,
    ) -> set:
        """Find topic boundaries from embedding similarity drops between
        consecutive sentences. No LLM needed — deterministic and cheap."""
        if embedding_manager is None or len(sentences) < 4:
            return set()
        try:
            vecs = embedding_manager.generate_embeddings(sentences)
        except Exception as e:  # noqa: BLE001
            print(f"Warning: semantic boundary detection failed ({e})")
            return set()

        boundaries: set = set()
        window = 3
        for i in range(1, len(sentences)):
            start = max(0, i - window)
            prev_mean = vecs[start:i].mean(axis=0)
            next_mean = vecs[i : min(len(vecs), i + window)].mean(axis=0)
            sim_prev = float(
                vecs[i - 1] @ prev_mean
                / (np.linalg.norm(vecs[i - 1]) * np.linalg.norm(prev_mean) + 1e-9)
            )
            sim_next = float(
                vecs[i] @ next_mean
                / (np.linalg.norm(vecs[i]) * np.linalg.norm(next_mean) + 1e-9)
            )
            if sim_prev - sim_next > 0.08:
                boundaries.add(i)
        return boundaries

    def split(self, text: str, embedding_manager=None) -> List[str]:
        """Split a prose run into topic-coherent pieces <= max_chunk_size."""
        sentences = self._split_into_sentences(text)
        if not sentences:
            return []
        if len(sentences) <= 2 and len(text) <= self.max_chunk_size:
            return [text]

        boundaries = set()
        if self.strategy in ("hybrid", "agentic"):
            boundaries = self._llm_boundaries(sentences)
            if not boundaries and self.strategy == "hybrid":
                boundaries = self._semantic_boundaries(sentences, embedding_manager)
        elif self.strategy == "semantic":
            boundaries = self._semantic_boundaries(sentences, embedding_manager)

        pieces: List[str] = []
        current: List[str] = []
        current_len = 0
        for idx, sentence in enumerate(sentences):
            s_len = len(sentence)
            if (idx in boundaries and current) or (
                current_len + s_len > self.max_chunk_size
            ):
                pieces.append(" ".join(current))
                current = [sentence]
                current_len = s_len
            else:
                current.append(sentence)
                current_len += s_len + 1
        if current:
            pieces.append(" ".join(current))
        return pieces


# ---------------------------------------------------------------------------
# Chunk packing
# ---------------------------------------------------------------------------


def _pack_units(
    units: List[Tuple[str, str]],
    splitter: TopicSplitter,
    embedding_manager=None,
    max_size: int = CHUNK_MAX_SIZE,
) -> List[Tuple[str, str]]:
    """Pack atomic units into chunks. Long prose runs are split by the
    TopicSplitter; code blocks and tables are never split."""
    packed: List[Tuple[str, str]] = []

    for kind, text in units:
        if kind == "prose" and len(text) > max_size:
            for piece in splitter.split(text, embedding_manager):
                if piece.strip():
                    packed.append(("prose", piece.strip()))
            continue

        if not packed:
            packed.append((kind, text))
            continue

        last_kind, last_text = packed[-1]
        # Never merge two code blocks (each stays an independent chunk),
        # and never merge a code block into a chunk that already has one.
        if kind == "code" and last_kind == "code":
            packed.append((kind, text))
        elif len(last_text) + len(text) + 1 <= max_size and not (
            kind == "code" or last_kind == "code"
        ):
            packed[-1] = (last_kind, last_text + "\n" + text)
        else:
            packed.append((kind, text))
    return packed


# ---------------------------------------------------------------------------
# Document-level chunking
# ---------------------------------------------------------------------------


def _section_path(section_metadata: dict) -> str:
    parts = []
    for key in ("endpoint", "block_type", "field_group"):
        value = section_metadata.get(key)
        if value:
            parts.append(value)
    return " / ".join(parts)


def _context_prefix(section_metadata: dict, kind: str) -> str:
    """Small self-containment header injected at the top of each chunk."""
    path = _section_path(section_metadata)
    if not path:
        return ""
    kind_label = {"code": "Example", "table": "Details", "prose": ""}.get(kind, "")
    label = f"{kind_label} - {path}" if kind_label else path
    return f"### {label}\n"


def _split_markdown_doc(
    doc: Document,
    splitter: TopicSplitter,
    embedding_manager=None,
    use_context_prefix: bool = CHUNK_CONTEXT_PREFIX,
) -> List[Document]:
    md_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=MARKDOWN_HEADERS_TO_SPLIT_ON,
        strip_headers=False,
    )
    sections = md_splitter.split_text(doc.page_content)

    chunks: List[Document] = []
    for section in sections:
        units = _segment_units(section.page_content)
        packed = _pack_units(units, splitter, embedding_manager)
        for kind, text in packed:
            if kind == "code" and len(text) >= 10:
                # Keep short code examples even below CHUNK_MIN_LENGTH.
                pass
            elif len(text.strip()) < CHUNK_MIN_LENGTH:
                continue

            content = text
            if use_context_prefix:
                content = _context_prefix(section.metadata, kind) + content

            merged_metadata = {
                **doc.metadata,
                **section.metadata,
                "kind": kind,
            }
            if _section_path(section.metadata):
                merged_metadata["section_path"] = _section_path(section.metadata)
            chunks.append(Document(page_content=content, metadata=merged_metadata))
    return chunks


def chunk_documents(
    documents: List[Document],
    embedding_manager=None,
) -> List[Document]:
    """Main entry point: apply structured agentic chunking to all documents."""
    strategy = CHUNK_STRATEGY
    print(f"Initializing chunker (strategy={strategy}, max_size={CHUNK_MAX_SIZE})...")
    splitter = TopicSplitter(strategy=strategy)
    all_chunks: List[Document] = []

    for i, doc in enumerate(documents):
        source = doc.metadata.get("source", f"Doc {i + 1}")
        print(f"Chunking document {i + 1}/{len(documents)}: {source}")
        doc_chunks = _split_markdown_doc(doc, splitter, embedding_manager)
        all_chunks.extend(doc_chunks)

    for i, chunk in enumerate(all_chunks):
        chunk.metadata["chunk_index"] = i
        chunk.metadata["content_length"] = len(chunk.page_content)

    print(f"Split {len(documents)} documents into {len(all_chunks)} chunks")
    return all_chunks