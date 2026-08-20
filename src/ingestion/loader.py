"""Loaders: raw files on disk -> LangChain Document objects.

Ingestion answers "what documents exist and where do they live";
chunking answers "how do we split them for embedding".
"""

from typing import List

from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_core.documents import Document

from ..config import PDF_DIR, TEXT_DIR
from .pdf_structure import load_pdfs_as_markdown_documents


def load_documents() -> List[Document]:
    """Load every .txt file from TEXT_DIR and every .pdf from PDF_DIR into a
    single flat list of Documents.

    PDFs are not loaded with a plain text loader — that throws away font
    size/bold info, which is the only signal a PDF gives about what's a
    heading vs. body text. They go through
    pdf_structure.load_pdfs_as_markdown_documents() instead, which
    reconstructs that structure as markdown-style headings.
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
    print(
        f"Loaded {len(documents)} raw documents "
        f"({TEXT_DIR} + {PDF_DIR})"
    )
    return documents