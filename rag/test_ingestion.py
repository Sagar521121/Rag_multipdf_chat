"""Runnable ingestion smoke tests: python -m rag.test_ingestion."""

from pathlib import Path

from rag.ingestion import process_pdf


def main() -> None:
    pdf = Path("data/PDFs/bert.pdf")
    chunks = process_pdf(pdf, source_name="bert.pdf")
    assert chunks, "ingestion must produce chunks"
    assert all(chunk["source"] == "bert.pdf" for chunk in chunks)
    assert all(chunk["page"] > 0 for chunk in chunks)
    assert all(chunk["text"].strip() for chunk in chunks)
    print(f"PASS ingestion: {len(chunks)} chunks")


if __name__ == "__main__":
    main()
