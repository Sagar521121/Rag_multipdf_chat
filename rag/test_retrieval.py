"""Runnable retrieval smoke test: python -m rag.test_retrieval."""

from rag.retrieval import retrieve


def main() -> None:
    results = retrieve("What is BERT?", top_k=3)
    assert set(results) == {"documents", "metadatas", "distances"}
    assert len(results["documents"][0]) == len(results["metadatas"][0])
    assert len(results["documents"][0]) == len(results["distances"][0])
    print(f"PASS retrieval: returned {len(results['documents'][0])} results")


if __name__ == "__main__":
    main()
