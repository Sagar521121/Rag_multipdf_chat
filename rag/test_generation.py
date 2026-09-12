"""Runnable generation/refusal tests without a network call."""

import rag.generation as generation
from rag.relevance import REFUSAL


class _Response:
    text = "BERT uses bidirectional representations. [bert.pdf, Page 1]"


class _Models:
    def generate_content(self, **kwargs):
        assert "BERT" in kwargs["contents"]
        return _Response()


class _Client:
    models = _Models()


def _results(text: str, distance: float = 0.5) -> dict:
    return {
        "documents": [[text]],
        "metadatas": [[{"source": "bert.pdf", "page": 1, "chunk_id": 0}]],
        "distances": [[distance]],
    }


def main() -> None:
    original_retrieve = generation.retrieve
    original_client = generation.get_gemini_client
    try:
        generation.retrieve = lambda query, top_k: _results("BERT is bidirectional.")
        generation.get_gemini_client = lambda: _Client()
        answer = generation.generate_answer("What is BERT?")
        assert answer.startswith("BERT uses")

        generation.retrieve = lambda query, top_k: _results("OpenAI GPT is mentioned.")
        refusal = generation.generate_answer("What is GPT-3?")
        assert refusal == REFUSAL
        print("PASS generation: grounded answer and controlled refusal")
    finally:
        generation.retrieve = original_retrieve
        generation.get_gemini_client = original_client


if __name__ == "__main__":
    main()
