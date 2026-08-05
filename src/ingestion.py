"""
INGESTION (a.k.a. "Data Injection")
------------------------------------
Responsible for getting raw content off disk and into LangChain
Document objects. This is deliberately separate from chunking:
ingestion answers "what documents exist and where do they live",
chunking answers "how do we split them for embedding".
"""

import os
from typing import List

from langchain_core.documents import Document
from langchain_community.document_loaders import DirectoryLoader, TextLoader

from config import TEXT_DIR, PDF_DIR
from pdf_structure import load_pdfs_as_markdown_documents


def make_sample_document() -> Document:
    """Example of constructing a Document manually (useful for testing)."""
    return Document(
        page_content="this is the environment that make u more effective",
        metadata={
            "source": "the Book",
            "Author": "VNY Sambangi",
            "pages": 3,
            "timestamp": "2026-08-04",
        },
    )


# `write_sample_text_files` removed per user request — ingestion only loads existing files


def load_documents() -> List[Document]:
    """
    Load every .txt file from TEXT_DIR and every .pdf from PDF_DIR
    into a single flat list of Document objects.

    PDFs are NOT loaded with a plain text loader — that throws away
    font size/bold info, which is the only signal a PDF gives you
    about what's a heading vs. body text. Instead they go through
    pdf_structure.load_pdfs_as_markdown_documents(), which reconstructs
    that structure as markdown-style headings so chunking.py can split
    on it the same way it would split a real .md file.
    """
    text_loader = DirectoryLoader(
        str(TEXT_DIR),
        glob=["**/*.txt"],
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
        show_progress=False,
    )

    text_documents = text_loader.load()
    pdf_documents = load_pdfs_as_markdown_documents(str(PDF_DIR))

    documents = text_documents + pdf_documents
    print(f"Loaded {len(documents)} raw documents "
          f"({TEXT_DIR} + {PDF_DIR})")
    return documents


if __name__ == "__main__":
    docs = load_documents()
    for d in docs:
        print(d.metadata.get("source"), "-", len(d.page_content), "chars")