"""Cached BGE document and query embeddings."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import streamlit as st
from sentence_transformers import SentenceTransformer


MODEL_NAME = "BAAI/bge-base-en-v1.5"
QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "


@st.cache_resource(show_spinner="Loading the embedding model...")
def load_embedding_model() -> SentenceTransformer:
    """Load BGE once per Streamlit process."""

    return SentenceTransformer(MODEL_NAME)


def create_embeddings(chunks: Sequence[dict[str, Any]]):
    """Create normalized embeddings for document chunks."""

    if not chunks:
        raise ValueError("Cannot create document embeddings for zero chunks")
    texts = [str(chunk["text"]) for chunk in chunks]
    return load_embedding_model().encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=False,
    )


def create_query_embedding(query: str):
    """Create one normalized BGE embedding using the retrieval instruction."""

    query = query.strip()
    if not query:
        raise ValueError("Cannot embed an empty query")
    return load_embedding_model().encode(
        QUERY_INSTRUCTION + query,
        normalize_embeddings=True,
        show_progress_bar=False,
    )


def embedding_dimension() -> int:
    """Return the configured model's vector dimension."""

    model = load_embedding_model()
    dimension = (
        model.get_embedding_dimension()
        if hasattr(model, "get_embedding_dimension")
        else model.get_sentence_embedding_dimension()
    )
    if dimension is None:
        raise RuntimeError("The embedding model did not expose its vector dimension")
    return int(dimension)
