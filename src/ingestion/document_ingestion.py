"""
Stage 10 of the Industrial AI MVP:
    Technical PDFs
          ↓
    page-aware text extraction
          ↓
    normalized document objects
          ↓
    document_catalog.json
    document_pages.jsonl

For the MVP we use PyMuPDF because it gives reliable page-level provenance.
Later, Docling can replace/augment this stage for richer layout/table/diagram
extraction without changing the downstream data contract.

Usage:
    uv add pymupdf

    uv run python src/ingestion/document_ingestion.py

Expected input:
    data/raw/refinery/documents/*.pdf

Outputs:
    data/processed/refinery/documents/
        document_catalog.json
        document_pages.jsonl
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import pymupdf

DEFAULT_INPUT_DIR = Path("data/raw/refinery/documents")
DEFAULT_OUTPUT_DIR = Path("data/processed/refinery/documents")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def normalize_text(text: str) -> str:
    """Normalize whitespace while preserving paragraph boundaries."""
    lines = [line.strip() for line in text.splitlines()]

    output: list[str] = []
    previous_blank = False

    for line in lines:
        if not line:
            if not previous_blank:
                output.append("")
            previous_blank = True
            continue

        output.append(line)
        previous_blank = False

    return "\n".join(output).strip()


def document_id(path: Path, sha256: str) -> str:
    value = f"{path.name}|{sha256}"
    return "doc_" + hashlib.sha1(value.encode("utf-8")).hexdigest()[:16]


def extract_pdf(
    path: Path,
    output_dir: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    sha256 = sha256_file(path)
    doc_id = document_id(path, sha256)

    pdf = pymupdf.open(path)

    document = {
        "document_id": doc_id,
        "file_name": path.name,
        "file_path": str(path.resolve()),
        "source_type": "pdf",
        "sha256": sha256,
        "page_count": pdf.page_count,
        "status": "parsed",
        "extraction_method": "PyMuPDF",
    }

    pages: list[dict[str, Any]] = []
    total_chars = 0

    for page_number in range(pdf.page_count):
        page = pdf.load_page(page_number)

        text = normalize_text(page.get_text("text"))

        total_chars += len(text)

        pages.append(
            {
                "document_id": doc_id,
                "file_name": path.name,
                "page": page_number + 1,
                "text": text,
                "char_count": len(text),
                "provenance": {
                    "source_id": doc_id,
                    "source_type": "pdf",
                    "location": {
                        "file": path.name,
                        "page": page_number + 1,
                    },
                    "extraction_method": "PyMuPDF",
                },
            }
        )

    pdf.close()

    document["total_characters"] = total_chars

    return document, pages


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract page-aware text from refinery technical PDFs."
    )

    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )

    args = parser.parse_args()

    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()

    if not input_dir.exists():
        raise FileNotFoundError(
            f"Document directory does not exist: {input_dir}\n"
            "Create it and put one or more technical PDFs inside."
        )

    pdf_files = sorted(input_dir.glob("*.pdf"))

    if not pdf_files:
        raise FileNotFoundError(
            f"No PDF files found in: {input_dir}\n"
            "Add a public technical/manual PDF before running this stage."
        )

    output_dir.mkdir(parents=True, exist_ok=True)

    catalog: list[dict[str, Any]] = []
    all_pages: list[dict[str, Any]] = []

    print("=" * 60)
    print("DOCUMENT INGESTION")
    print("=" * 60)
    print(f"Input directory : {input_dir}")
    print(f"PDF files       : {len(pdf_files)}")
    print()

    for pdf_path in pdf_files:
        print(f"Processing: {pdf_path.name}")

        document, pages = extract_pdf(
            path=pdf_path,
            output_dir=output_dir,
        )

        catalog.append(document)
        all_pages.extend(pages)

        print(
            f"  pages={document['page_count']} "
            f"characters={document['total_characters']}"
        )

    catalog_path = output_dir / "document_catalog.json"

    with catalog_path.open("w", encoding="utf-8") as f:
        json.dump(
            catalog,
            f,
            indent=2,
            ensure_ascii=False,
        )

    pages_path = output_dir / "document_pages.jsonl"

    with pages_path.open("w", encoding="utf-8") as f:
        for page in all_pages:
            f.write(
                json.dumps(
                    page,
                    ensure_ascii=False,
                )
                + "\n"
            )

    print()
    print(f"Created: {catalog_path}")
    print(f"Created: {pages_path}")
    print(f"Documents: {len(catalog)}")
    print(f"Pages: {len(all_pages)}")
    print()
    print(
        "Next stage: chunk document pages and build retrieval-ready "
        "document evidence."
    )
    print("=" * 60)


if __name__ == "__main__":
    main()
