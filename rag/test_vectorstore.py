"""Runnable vectorstore lifecycle tests: python -m rag.test_vectorstore."""

from rag.vectorstore import collection, add_chunks, delete_document


def main() -> None:
    source = "vectorstore-test.pdf"
    try:
        first = [
            {"source": source, "page": 1, "chunk_id": 0, "text": "old one"},
            {"source": source, "page": 1, "chunk_id": 1, "text": "old two"},
        ]
        vectors = [[0.0] * 768, [0.0] * 768]
        assert add_chunks(first, vectors) == 2
        second = [{"source": source, "page": 1, "chunk_id": 0, "text": "new one"}]
        assert add_chunks(second, [[0.0] * 768]) == 1
        current = collection.get(where={"source": source}, include=["documents"])
        assert current["documents"] == ["new one"]
        assert delete_document(source) == 1
        assert not collection.get(where={"source": source}, include=[])["ids"]
        print("PASS vectorstore: re-index and delete remove stale vectors")
    finally:
        delete_document(source)


if __name__ == "__main__":
    main()
