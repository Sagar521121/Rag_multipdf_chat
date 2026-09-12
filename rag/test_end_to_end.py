"""Runnable offline end-to-end RAG test: python -m rag.test_end_to_end."""

from pathlib import Path

import rag.generation as generation
from rag.embeddings import create_embeddings
from rag.ingestion import process_pdf
from rag.vectorstore import add_chunks, delete_document
from rag.retrieval import retrieve


class _Response:
    text = "BERT is described in the indexed context. [e2e-bert.pdf, Page 1]"


class _Models:
    def generate_content(self, **kwargs):
        return _Response()


class _Client:
    models = _Models()


def main() -> None:
    source = "e2e-bert.pdf"
    original_client = generation.get_gemini_client
    try:
        chunks = process_pdf(Path("data/PDFs/bert.pdf"), source_name=source)
        add_chunks(chunks, create_embeddings(chunks))
        results = retrieve("What is BERT?", top_k=3)
        assert results["documents"][0]
        generation.get_gemini_client = lambda: _Client()
        details = generation.generate_answer("What is BERT?", return_details=True)
        assert details["answer"]
        assert details["relevant"]
        print(f"PASS end-to-end: {len(chunks)} chunks indexed and queried")
    finally:
        generation.get_gemini_client = original_client
        delete_document(source)


if __name__ == "__main__":
    main()
