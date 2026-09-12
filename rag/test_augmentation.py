"""Runnable context construction test: python -m rag.test_augmentation."""

from rag.augmentation import build_context


def main() -> None:
    results = {
        "documents": [["BERT text", "BERT text"]],
        "metadatas": [[
            {"source": "bert.pdf", "page": 3, "chunk_id": 2},
            {"source": "bert.pdf", "page": 3, "chunk_id": 2},
        ]],
        "distances": [[0.5, 0.5]],
    }
    context = build_context(results)
    assert context.count("bert.pdf") == 1
    assert "Page: 3" in context and "Chunk: 2" in context
    print("PASS augmentation: source/page/chunk attribution and deduplication")


if __name__ == "__main__":
    main()
