"""PDF extraction, filtering, page grouping, and page-aware chunking."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader
from unstructured.partition.pdf import partition_pdf


class PDFExtractionError(RuntimeError):
    """Raised when a PDF cannot produce usable text through either extractor."""


@dataclass
class FallbackElement:
    """Unstructured-compatible element used for pypdf fallback text."""

    text: str
    page_number: int
    element_type: str = "NarrativeText"

    @property
    def metadata(self) -> dict[str, int]:
        return {"page_number": self.page_number}


def _element_text(element: Any) -> str:
    return str(getattr(element, "text", None) or "").strip()


def _page_number(element: Any) -> int | None:
    metadata = getattr(element, "metadata", None)
    page = getattr(metadata, "page_number", None)
    if page is None and isinstance(metadata, dict):
        page = metadata.get("page_number")
    return int(page) if page is not None else None


def _element_type(element: Any) -> str:
    return getattr(element, "element_type", None) or type(element).__name__


def _usable(elements: list[Any] | None) -> list[Any]:
    return [element for element in (elements or []) if _element_text(element)]


def _extract_with_pypdf(pdf_path: str | Path) -> list[FallbackElement]:
    try:
        reader = PdfReader(str(pdf_path))
        elements: list[FallbackElement] = []
        for page_number, page in enumerate(reader.pages, start=1):
            text = (page.extract_text() or "").strip()
            if text:
                elements.append(FallbackElement(text=text, page_number=page_number))
        return elements
    except Exception as exc:
        raise PDFExtractionError(f"pypdf extraction failed for {pdf_path}: {exc}") from exc


def extract_elements(pdf_path: str | Path) -> list[Any]:
    """Use Unstructured fast first, then pypdf; never return an empty success."""

    path = Path(pdf_path)
    fast_error: Exception | None = None
    try:
        elements = _usable(
            partition_pdf(filename=str(path), strategy="fast", languages=["eng"])
        )
        if elements:
            return elements
        fast_error = ValueError("Unstructured fast extraction returned no usable elements")
    except Exception as exc:
        fast_error = exc

    fallback_elements = _usable(_extract_with_pypdf(path))
    if fallback_elements:
        return fallback_elements

    detail = f"Unstructured fast: {fast_error}; pypdf: no usable text"
    raise PDFExtractionError(f"No text could be extracted from {path}. {detail}")


def filter_elements(elements: list[Any]) -> list[Any]:
    """Remove empty elements and Unstructured footer elements."""

    return [
        element
        for element in elements
        if _element_text(element) and _element_type(element) != "Footer"
    ]


def create_documents(
    filtered_elements: list[Any],
    pdf_path: str | Path,
    source_name: str | None = None,
) -> list[Document]:
    """Convert extracted elements into LangChain Documents."""

    source = source_name or Path(pdf_path).name
    documents: list[Document] = []
    for element in filtered_elements:
        page_number = _page_number(element)
        if page_number is None:
            continue
        documents.append(
            Document(
                page_content=_element_text(element),
                metadata={
                    "source": source,
                    "page": page_number,
                    "element_type": _element_type(element),
                },
            )
        )
    return documents


def group_by_page(documents: list[Document]) -> dict[int, list[str]]:
    pages: dict[int, list[str]] = defaultdict(list)
    for document in documents:
        page_number = document.metadata.get("page")
        if page_number is not None and document.page_content.strip():
            pages[int(page_number)].append(document.page_content.strip())
    return dict(pages)


def create_chunks(
    pages: dict[int, list[str]],
    source_name: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> list[dict[str, Any]]:
    """Split each page independently so chunks never cross page boundaries."""

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    chunks: list[dict[str, Any]] = []
    for page_number in sorted(pages):
        page_text = "\n".join(pages[page_number]).strip()
        for chunk_id, chunk_text in enumerate(splitter.split_text(page_text)):
            text = chunk_text.strip()
            if text:
                chunks.append(
                    {
                        "source": source_name,
                        "page": page_number,
                        "chunk_id": chunk_id,
                        "text": text,
                    }
                )
    return chunks


def process_pdf(pdf_path: str | Path, source_name: str | None = None) -> list[dict[str, Any]]:
    """Run extraction through page-aware chunking for one PDF."""

    source = source_name or Path(pdf_path).name
    elements = filter_elements(extract_elements(pdf_path))
    if not elements:
        raise PDFExtractionError(f"No usable elements remain after filtering {pdf_path}")
    documents = create_documents(elements, pdf_path, source_name=source)
    pages = group_by_page(documents)
    if not pages:
        raise PDFExtractionError(f"No pages with usable text were created for {source}")
    chunks = create_chunks(pages, source)
    if not chunks:
        raise PDFExtractionError(f"No chunks were created for {source}")
    return chunks


def process_multiple_pdfs(
    pdf_paths: list[str | Path],
    source_names: dict[str | Path, str] | None = None,
) -> list[dict[str, Any]]:
    """Process multiple PDFs while retaining their original display names."""

    all_chunks: list[dict[str, Any]] = []
    for pdf_path in pdf_paths:
        source_name = None
        if source_names:
            source_name = source_names.get(pdf_path) or source_names.get(str(pdf_path))
        all_chunks.extend(process_pdf(pdf_path, source_name=source_name))
    if not all_chunks:
        raise PDFExtractionError("No usable chunks were created from the uploaded PDFs")
    return all_chunks
