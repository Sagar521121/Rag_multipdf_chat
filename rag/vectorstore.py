"""Persistent Chroma storage and document lifecycle operations."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import chromadb


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CHROMA_PATH = PROJECT_ROOT / "chroma_db"
COLLECTION_NAME = "research_papers"
COLLECTION_METADATA = {
    "embedding_model": "BAAI/bge-base-en-v1.5",
    "distance_metric": "l2_normalized_embeddings",
}

chroma_client = chromadb.PersistentClient(path=str(CHROMA_PATH))
collection = chroma_client.get_or_create_collection(
    name=COLLECTION_NAME,
    metadata=COLLECTION_METADATA,
)
if collection.metadata != COLLECTION_METADATA:
    collection.modify(metadata=COLLECTION_METADATA)


def _chunk_id(chunk: dict[str, Any]) -> str:
    source = str(chunk["source"])
    return f"{source}::page-{int(chunk['page'])}::chunk-{int(chunk['chunk_id'])}"


def delete_document(source: str) -> int:
    """Delete every vector belonging to one source filename."""

    existing = collection.get(where={"source": source}, include=[])
    ids = existing.get("ids", [])
    if ids:
        collection.delete(ids=ids)
    return len(ids)


def add_chunks(chunks: list[dict[str, Any]], embeddings) -> int:
    """Safely replace all chunks for each source, then insert the new version."""

    if not chunks:
        raise ValueError("Cannot index zero chunks")
    if len(chunks) != len(embeddings):
        raise ValueError(
            f"Chunk/embedding count mismatch: {len(chunks)} chunks, {len(embeddings)} embeddings"
        )

    sources = {str(chunk["source"]) for chunk in chunks}
    for source in sources:
        if not source or Path(source).name != source:
            raise ValueError(f"Chunk source must be an original filename, got: {source!r}")
        delete_document(source)

    ids = [_chunk_id(chunk) for chunk in chunks]
    documents = [str(chunk["text"]) for chunk in chunks]
    metadatas = [
        {
            "source": str(chunk["source"]),
            "page": int(chunk["page"]),
            "chunk_id": int(chunk["chunk_id"]),
        }
        for chunk in chunks
    ]
    vectors = [
        embedding.tolist() if hasattr(embedding, "tolist") else list(embedding)
        for embedding in embeddings
    ]
    collection.upsert(
        ids=ids,
        documents=documents,
        embeddings=vectors,
        metadatas=metadatas,
    )
    return len(ids)


def get_document_sources() -> list[str]:
    """Return only documents actually present in the persistent collection."""

    results = collection.get(include=["metadatas"])
    return sorted(
        {
            str(metadata["source"])
            for metadata in results.get("metadatas", [])
            if metadata and metadata.get("source")
        }
    )


def get_collection_count() -> int:
    return collection.count()


def get_collection_stats() -> dict[str, int]:
    """Return inspectable index statistics for the Streamlit UI."""

    results = collection.get(include=["metadatas"])
    metadatas = [metadata for metadata in results.get("metadatas", []) if metadata]
    sources = {metadata.get("source") for metadata in metadatas}
    pages = {(metadata.get("source"), metadata.get("page")) for metadata in metadatas}
    dimension = 0
    if collection.count():
        peek = collection.peek(limit=1)
        vectors = peek.get("embeddings")
        if vectors is not None and len(vectors):
            dimension = len(vectors[0])
    return {
        "documents": len({source for source in sources if source}),
        "pages": len({page for page in pages if page[0] is not None}),
        "chunks": len(metadatas),
        "vectors": collection.count(),
        "embedding_dimension": dimension,
    }
