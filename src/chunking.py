"""
CHUNKING
--------
Your OEM API docs are PDFs — no markdown/HTML tags. But
ingestion.load_documents() now runs every PDF through
pdf_structure.py first, which reads PyMuPDF's font-size/bold info and
reconstructs headings as markdown-style "#"/"##"/"###" lines. By the
time a document reaches this module, .pdf-sourced text already LOOKS
like markdown — so it's split the same way a real .md file would be:
on structure first (endpoint / Request / Response / Headers, etc.),
falling back to character-based splitting only for any individual
section that's still too large to embed cleanly.

  - .pdf and .md sources -> MarkdownHeaderTextSplitter
  - everything else (.txt, etc.) -> RecursiveCharacterTextSplitter only

Whichever path a document takes, every resulting chunk keeps:
  - the header trail it came from (e.g. endpoint="SetVolume",
    block_type="Response") as metadata, so retrieval can filter/boost
    on it later
  - the original document metadata (source, etc.)
  - a chunk_index for traceability
"""

from pathlib import Path
from typing import List

from langchain_core.documents import Document
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)

DEFAULT_CHUNK_SIZE = 800
DEFAULT_CHUNK_OVERLAP = 100

# Adjust these to match how your OEM docs actually mark up sections.
# Example assumed structure per endpoint:
#   # SetVolume
#   ## Request
#   ## Response
#   ### Headers
MARKDOWN_HEADERS_TO_SPLIT_ON = [
    ("#", "endpoint"),
    ("##", "block_type"),      # Request / Response / Headers / Errors
    ("###", "field_group"),
]

def _resplit_if_oversized(text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
    """A header-defined section can still be long (e.g. a big JSON
    schema). Only re-split with the character splitter if needed —
    short sections pass through untouched so a whole Request/Response
    block stays intact whenever it fits."""
    if len(text) <= chunk_size:
        return [text]
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_text(text)


def _split_markdown_doc(
    doc: Document, chunk_size: int, chunk_overlap: int
) -> List[Document]:
    splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=MARKDOWN_HEADERS_TO_SPLIT_ON,
        strip_headers=False,  # keep header text in the chunk itself
    )
    sections = splitter.split_text(doc.page_content)

    chunks: List[Document] = []
    for section in sections:
        for piece in _resplit_if_oversized(section.page_content, chunk_size, chunk_overlap):
            merged_metadata = {**doc.metadata, **section.metadata}
            chunks.append(Document(page_content=piece, metadata=merged_metadata))
    return chunks


def _split_plain_doc(
    doc: Document, chunk_size: int, chunk_overlap: int
) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_documents([doc])


def chunk_documents(
    documents: List[Document],
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> List[Document]:
    """
    Route each document to the right splitter based on its file
    extension, then tag every chunk with a chunk_index.
    """
    all_chunks: List[Document] = []

    for doc in documents:
        source = str(doc.metadata.get("source", ""))
        ext = Path(source).suffix.lower()

        if ext in (".md", ".pdf"):
            # .pdf already arrives as markdown-style text — see
            # pdf_structure.py, run during ingestion.
            doc_chunks = _split_markdown_doc(doc, chunk_size, chunk_overlap)
        else:
            # .txt or anything without reconstructed markup — no
            # header structure to key off, so fall back to
            # character-based splitting.
            doc_chunks = _split_plain_doc(doc, chunk_size, chunk_overlap)

        all_chunks.extend(doc_chunks)

    for i, chunk in enumerate(all_chunks):
        chunk.metadata["chunk_index"] = i
        chunk.metadata["content_length"] = len(chunk.page_content)

    print(
        f"Split {len(documents)} documents into {len(all_chunks)} chunks "
        f"(chunk_size={chunk_size}, overlap={chunk_overlap})"
    )
    return all_chunks


if __name__ == "__main__":
    from ingestion import load_documents

    docs = load_documents()
    chunks = chunk_documents(docs)
    for c in chunks[:5]:
        header_trail = " > ".join(
            str(c.metadata[k]) for k in ("endpoint", "block_type", "field_group") if k in c.metadata
        )
        print(f"[{header_trail or 'no headers'}] {c.page_content[:80]!r}...")