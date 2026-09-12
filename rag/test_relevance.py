"""Runnable relevance/refusal tests: python -m rag.test_relevance."""

from rag.relevance import REFUSAL, has_relevant_results


def _results(text: str, distance: float) -> dict:
    return {
        "documents": [[text]],
        "metadatas": [[{"source": "bert.pdf", "page": 1, "chunk_id": 0}]],
        "distances": [[distance]],
    }


def main() -> None:
    assert has_relevant_results(_results("BERT is a language model.", 0.60), "What is BERT?")
    assert not has_relevant_results(_results("OpenAI GPT is mentioned.", 0.60), "What is GPT-3?")
    assert not has_relevant_results(_results("BERT is a language model.", 1.20), "What is BERT?")
    assert REFUSAL.startswith("I don't have enough information")
    print("PASS relevance: supported, version-mismatch, and distant queries")


if __name__ == "__main__":
    main()
