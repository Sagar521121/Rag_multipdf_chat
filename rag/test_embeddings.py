"""Runnable embedding smoke tests: python -m rag.test_embeddings."""

from rag.embeddings import create_embeddings, create_query_embedding, embedding_dimension


def main() -> None:
    chunks = [{"text": "BERT uses bidirectional Transformer representations."}]
    document_vectors = create_embeddings(chunks)
    query_vector = create_query_embedding("What is BERT?")
    assert len(document_vectors) == 1
    assert len(document_vectors[0]) == embedding_dimension()
    assert len(query_vector) == len(document_vectors[0])
    print(f"PASS embeddings: dimension={len(query_vector)}")


if __name__ == "__main__":
    main()
