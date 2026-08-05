"""
PDF STRUCTURE RECONSTRUCTION
----------------------------
PDFs have no markdown/HTML tags — PyMuPDFLoader (what you were using)
just dumps flat text and throws away font size/bold/position info in
the process. That's exactly the information that tells you "this line
is a heading" in a PDF.

This module reads PDFs directly with PyMuPDF (fitz) at the span level
and reconstructs headings from THREE signals, in priority order:

  1. Numbered sections, e.g. "3.1 NULL command" or "2.5.1 Numeric
     value parameters" — very common in hardware/protocol command
     references like RTI driver API docs. Heading depth = number of
     dot-separated parts (1 -> #, 2 -> ##, 3+ -> ###).
  2. Known API-doc section labels by name (Request, Response, Headers,
     Parameters, Body, Errors, etc.) even when their font isn't
     visually bigger than body text.
  3. Font size rank (largest 2 distinct sizes in the doc -> title/#,
     subtitle/##) — mainly catches the cover-page title.

It also actively REMOVES two kinds of noise before any of the above
runs:
  - Repeated running headers/footers (e.g. a page number or doc
    number that appears on nearly every page) — detected by how many
    distinct pages a line shows up on, not by content, so it works
    regardless of what your footer actually says.
  - Table-of-contents dot-leader lines ("Data format ..... 4") — these
    duplicate real headings found later in the body and just add
    noise chunks with no useful content.

The output is a single markdown-style string per PDF, which
chunking.py's MarkdownHeaderTextSplitter then splits the same way it
would split a real .md file.

This is a heuristic — inspect the output on 2-3 of your real docs (see
the __main__ block below) before trusting it on all of them.
"""

import re
from collections import Counter
from pathlib import Path
from typing import List

import fitz  # PyMuPDF
from langchain_core.documents import Document

# Section labels common in API/protocol documentation. Lines matching
# these (case-insensitive, short lines only) are treated as headers
# even if their font size doesn't stand out from body text.
KEYWORD_HEADERS = {
    "request", "response", "endpoint", "endpoints", "parameters",
    "query parameters", "path parameters", "headers", "body",
    "request body", "response body", "errors", "error codes",
    "example request", "example response", "authentication",
    "rate limits", "status codes", "description",
}

# "3.1 NULL command" / "2.5.1 Numeric value parameters" / "1 OUTLINE"
# Depth = number of dot-separated segments in the leading number.
NUMBERED_HEADING_RE = re.compile(
    r"^(?P<num>\d{1,2}(\.\d{1,2}){0,3})\s+(?P<title>[A-Za-z][\w\-/,'()& ]{1,80})$"
)

# TOC dot-leader lines: "Data format ..................... 4"
TOC_DOT_LEADER_RE = re.compile(r"\.{3,}\s*\d+\s*$")

# A bare page-number-ish line: "2 / 45", "Page 3", "12"
PAGE_NUMBER_RE = re.compile(r"^(page\s*)?\d+(\s*/\s*\d+)?$", re.IGNORECASE)


def _line_is_heading_keyword(text: str) -> bool:
    normalized = text.strip().lower().rstrip(":")
    return normalized in KEYWORD_HEADERS and len(text.split()) <= 4


def _numbered_heading_match(text: str):
    m = NUMBERED_HEADING_RE.match(text.strip())
    if not m:
        return None
    # Reject if the "title" part still has a trailing TOC page number
    # e.g. "3.1 NULL command .......... 9" would already be caught by
    # TOC_DOT_LEADER_RE upstream, but guard here too.
    if TOC_DOT_LEADER_RE.search(text):
        return None
    depth = m.group("num").count(".") + 1
    level = min(depth, 3)
    return level, text.strip()


def _collect_lines(doc) -> List[dict]:
    """One entry per visual line, with size/bold info and the page it's on."""
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
    """
    A line that shows up on a large fraction of distinct pages is a
    running header/footer (doc number, page number, "Confidential",
    etc.), not real content. Threshold: appears on >20% of pages
    (min 3 pages) — content coincidentally repeating 1-2 times won't
    trip this, but per-page furniture will.
    """
    pages_per_text: dict = {}
    for line in lines:
        pages_per_text.setdefault(line["text"], set()).add(line["page"])

    threshold = max(3, int(total_pages * 0.2))
    return {
        text for text, pages in pages_per_text.items()
        if len(pages) >= threshold
    }


def extract_pdf_as_markdown(pdf_path: str) -> str:
    """
    Read one PDF and return a single markdown-style string, with
    detected headings prefixed by #, ##, ### and noise (repeated
    headers/footers, TOC dot-leader lines) stripped out.
    """
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
    """
    Convert every .pdf in pdf_dir into ONE Document each (whole file,
    not split per page — sections shouldn't get cut just because a
    page boundary happens to fall in the middle of a Response block),
    with page_content already reformatted as markdown-style text.
    """
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


if __name__ == "__main__":
    from config import PDF_DIR

    docs = load_pdfs_as_markdown_documents(str(PDF_DIR))
    if not docs:
        print(f"No PDFs found in {PDF_DIR}")
    else:
        first = docs[0].page_content.splitlines()
        headings = [l for l in first if l.startswith("#")]

        print(f"\n--- {docs[0].metadata['source']} ---")
        print(f"Total lines: {len(first)} | Detected headings: {len(headings)}")

        print("\n--- First 25 lines ---")
        print("\n".join(first[:25]))

        print("\n--- All detected headings (first 40) ---")
        print("\n".join(headings[:40]))