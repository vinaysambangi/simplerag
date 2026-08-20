"""PDF structure reconstruction.

PDFs have no markdown/HTML tags — a plain text loader dumps flat text and
throws away font size / bold / position info, which is exactly what tells
you "this line is a heading" in a PDF.

This module reads PDFs directly with PyMuPDF at the span level and
reconstructs headings from three signals, in priority order:

  1. Numbered sections, e.g. "3.1 NULL command" or "2.5.1 Numeric value
     parameters". Heading depth = number of dot-separated parts.
  2. Known API-doc section labels by name (Request, Response, Headers,
     Parameters, Body, Errors, ...) even when their font isn't bigger.
  3. Font size rank (largest sizes in the doc -> title/subtitle).

It also strips two kinds of noise:
  - repeated running headers/footers (detected by page-frequency),
  - table-of-contents dot-leader lines.

Output is one markdown-style string per PDF, which the chunker splits
the same way it would split a real .md file.
"""

import re
from pathlib import Path
from typing import List

import fitz  # PyMuPDF
from langchain_core.documents import Document

KEYWORD_HEADERS = {
    "request", "response", "endpoint", "endpoints", "parameters",
    "query parameters", "path parameters", "headers", "body",
    "request body", "response body", "errors", "error codes",
    "example request", "example response", "authentication",
    "rate limits", "status codes", "description",
}

NUMBERED_HEADING_RE = re.compile(
    r"^(?P<num>\d{1,2}(\.\d{1,2}){0,3})\s+(?P<title>[A-Za-z][\w\-/,'()& ]{1,80})$"
)

TOC_DOT_LEADER_RE = re.compile(r"\.{3,}\s*\d+\s*$")

PAGE_NUMBER_RE = re.compile(r"^(page\s*)?\d+(\s*/\s*\d+)?$", re.IGNORECASE)


def _line_is_heading_keyword(text: str) -> bool:
    normalized = text.strip().lower().rstrip(":")
    return normalized in KEYWORD_HEADERS and len(text.split()) <= 4


def _numbered_heading_match(text: str):
    m = NUMBERED_HEADING_RE.match(text.strip())
    if not m:
        return None
    if TOC_DOT_LEADER_RE.search(text):
        return None
    depth = m.group("num").count(".") + 1
    return min(depth, 3), text.strip()


def _collect_lines(doc) -> List[dict]:
    """One entry per visual line, with size info and the page it's on."""
    lines = []
    for page_num, page in enumerate(doc):
        blocks = page.get_text("dict")["blocks"]
        for block in blocks:
            if "lines" not in block:
                continue
            for line in block["lines"]:
                spans = line["spans"]
                text = "".join(s["text"] for s in spans).strip()
                if not text:
                    continue
                max_size = round(max(s["size"] for s in spans), 1)
                lines.append({"text": text, "size": max_size, "page": page_num})
    return lines


def _find_repeated_header_footer_lines(lines: List[dict], total_pages: int) -> set:
    """Lines appearing on a large fraction of pages are running furniture."""
    pages_per_text: dict = {}
    for line in lines:
        pages_per_text.setdefault(line["text"], set()).add(line["page"])

    threshold = max(3, int(total_pages * 0.2))
    return {
        text for text, pages in pages_per_text.items()
        if len(pages) >= threshold
    }


def extract_pdf_as_markdown(pdf_path: str) -> str:
    """Read one PDF and return a markdown-style string with headings marked."""
    doc = fitz.open(pdf_path)
    total_pages = doc.page_count
    lines = _collect_lines(doc)
    doc.close()

    noise_lines = _find_repeated_header_footer_lines(lines, total_pages)
    distinct_sizes = sorted({l["size"] for l in lines}, reverse=True)
    size_to_level = {size: i + 1 for i, size in enumerate(distinct_sizes[:2])}

    out_lines = []
    for line in lines:
        text = line["text"]

        if text in noise_lines:
            continue
        if TOC_DOT_LEADER_RE.search(text):
            continue
        if PAGE_NUMBER_RE.match(text):
            continue

        numbered = _numbered_heading_match(text)
        size_level = size_to_level.get(line["size"])

        if numbered:
            level, title = numbered
            out_lines.append(f"{'#' * level} {title}")
        elif size_level:
            out_lines.append(f"{'#' * size_level} {text}")
        elif _line_is_heading_keyword(text):
            out_lines.append(f"## {text}")
        else:
            out_lines.append(text)

    return "\n".join(out_lines)


def load_pdfs_as_markdown_documents(pdf_dir: str) -> List[Document]:
    """Convert every .pdf in pdf_dir into one Document (whole file, not per
    page, so sections don't get cut at page boundaries)."""
    documents = []
    for pdf_path in sorted(Path(pdf_dir).glob("*.pdf")):
        markdown_text = extract_pdf_as_markdown(str(pdf_path))
        documents.append(
            Document(
                page_content=markdown_text,
                metadata={"source": str(pdf_path)},
            )
        )
    print(f"Converted {len(documents)} PDFs to structured markdown text")
    return documents