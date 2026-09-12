"""Query embedding and inspectable Chroma retrieval."""

from __future__ import annotations

from rag.embeddings import create_query_embedding
from rag.vectorstore import collection


def retrieve(query: str, top_k: int = 12) -> dict:
    """
    Retrieve a larger candidate pool from Chroma.

    The caller can later select/rerank the strongest chunks.
    We retrieve more candidates than the final context size so that
    useful evidence from multiple documents has a chance to be selected.
    """

    if collection.count() == 0:
        return {
            "documents": [[]],
            "metadatas": [[]],
            "distances": [[]],
        }

    if top_k < 1:
        raise ValueError("top_k must be at least 1")

    query_embedding = create_query_embedding(query)

    results = collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=min(top_k, collection.count()),
        include=[
            "documents",
            "metadatas",
            "distances",
        ],
    )

    return {
        "documents": results.get("documents", [[]]),
        "metadatas": results.get("metadatas", [[]]),
        "distances": results.get("distances", [[]]),
    }