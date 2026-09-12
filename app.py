"""Streamlit UI for the Multi-PDF RAG research assistant."""

from __future__ import annotations

import hashlib
import tempfile
from pathlib import Path

import streamlit as st

from rag.embeddings import create_embeddings
from rag.generation import generate_answer
from rag.ingestion import PDFExtractionError, process_multiple_pdfs
from rag.vectorstore import (
    collection,
    delete_document,
    get_collection_stats,
    get_document_sources,
)
from rag.vectorstore import add_chunks


st.set_page_config(
    page_title="Multi-PDF Research Assistant",
    page_icon=":material/menu_book:",
    layout="wide",
    initial_sidebar_state="expanded",
)


def _file_key(uploaded_file) -> str:
    digest = hashlib.sha256(uploaded_file.getvalue()).hexdigest()[:16]
    return f"{uploaded_file.name}:{uploaded_file.size}:{digest}"


def _sources_from_results(results: dict) -> list[dict]:
    documents = (results.get("documents") or [[]])[0]
    metadatas = (results.get("metadatas") or [[]])[0]
    distances = (results.get("distances") or [[]])[0]
    return [
        {
            "source": metadata.get("source", "unknown"),
            "page": metadata.get("page", "?"),
            "chunk_id": metadata.get("chunk_id", "?"),
            "distance": float(distances[index]) if index < len(distances) else None,
            "preview": str(documents[index])[:280].replace("\n", " "),
        }
        for index, metadata in enumerate(metadatas)
    ]


def _render_sources(sources: list[dict]) -> None:
    if not sources:
        return
    with st.expander("Inspect retrieved evidence", expanded=False):
        for rank, source in enumerate(sources, start=1):
            distance = source["distance"]
            distance_text = f" · distance {distance:.3f}" if distance is not None else ""
            st.markdown(
                f"**{rank}. {source['source']} · Page {source['page']} · "
                f"Chunk {source['chunk_id']}**{distance_text}"
            )
            st.caption(source["preview"])


def _render_sidebar() -> list:
    with st.sidebar:
        st.header("Documents")
        uploaded_files = st.file_uploader(
            "Choose PDF files",
            type=["pdf"],
            accept_multiple_files=True,
            key="pdf_uploader",
            help="Selecting files does not index them. Use BUILD INDEX when ready.",
        ) or []

        st.session_state.setdefault("excluded_uploads", set())
        selected_files = []
        selected_file_keys = set()
        for file in uploaded_files:
            file_key = _file_key(file)
            if file_key in st.session_state.excluded_uploads:
                continue
            if file_key in selected_file_keys:
                continue
            selected_file_keys.add(file_key)
            selected_files.append(file)

        st.subheader("Selected PDFs")
        if selected_files:
            for file in selected_files:
                file_key = _file_key(file)
                row = st.container(horizontal=True, vertical_alignment="center")
                row.write(f"{file.name} ({file.size / 1024:.0f} KB)")
                if row.button("Remove", key=f"remove_selection_{file_key}"):
                    st.session_state.excluded_uploads.add(file_key)
                    st.rerun()
        else:
            st.caption("No PDFs selected.")

        build_index = st.button(
            "Build index",
            type="primary",
            width="stretch",
            disabled=not selected_files,
            key="build_index",
        )

        st.divider()
        st.subheader("Indexed documents")
        indexed_sources = get_document_sources()
        if indexed_sources:
            for source in indexed_sources:
                row = st.container(horizontal=True, vertical_alignment="center")
                row.write(source)
                if row.button("Remove", key=f"remove_indexed_{source}"):
                    deleted = delete_document(source)
                    st.session_state.index_status = f"Removed {source} ({deleted} chunks)."
                    st.rerun()
        else:
            st.caption("No indexed documents.")

        stats = get_collection_stats()
        st.divider()
        st.subheader("Index status")
        st.success("Index ready" if stats["vectors"] else "No index")
        metric_columns = st.columns(2)
        metric_columns[0].metric("Documents", stats["documents"])
        metric_columns[1].metric("Pages", stats["pages"])
        metric_columns = st.columns(2)
        metric_columns[0].metric("Chunks", stats["chunks"])
        metric_columns[1].metric("Vectors", stats["vectors"])
        st.caption(f"Embedding dimension: {stats['embedding_dimension'] or 'not loaded'}")

    return selected_files if build_index else []


def _index_selected_files(selected_files: list) -> None:
    temporary_paths: list[Path] = []
    try:
        with st.spinner("Extracting, chunking, embedding, and indexing..."):
            source_names: dict[str, str] = {}
            with tempfile.TemporaryDirectory(prefix="rag_upload_") as temp_dir:
                for file in selected_files:
                    temp_path = Path(temp_dir) / file.name
                    temp_path.write_bytes(file.getvalue())
                    temporary_paths.append(temp_path)
                    source_names[str(temp_path)] = file.name

                chunks = process_multiple_pdfs(
                    [str(path) for path in temporary_paths],
                    source_names=source_names,
                )
                embeddings = create_embeddings(chunks)
                stored = add_chunks(chunks, embeddings)

        st.session_state.index_status = (
            f"Index ready: {stored} chunks from {len(selected_files)} document(s)."
        )
        st.success(st.session_state.index_status)
    except (PDFExtractionError, ValueError, RuntimeError) as exc:
        st.error(f"Indexing failed: {exc}")
    except Exception as exc:
        st.error(f"Unexpected indexing failure: {exc}")


def main() -> None:
    st.title("Multi-PDF Research Assistant")
    st.caption("Build a grounded research index, then ask questions with inspectable sources.")

    st.session_state.setdefault("messages", [])
    st.session_state.setdefault("index_status", "Select PDFs, then build an index.")

    files_to_index = _render_sidebar()
    if files_to_index:
        _index_selected_files(files_to_index)

    if st.session_state.index_status:
        st.info(st.session_state.index_status)

    st.subheader("Conversation")
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message.get("sources"):
                _render_sources(message["sources"])

    query = st.chat_input("Ask about your indexed PDFs")
    if not query:
        return
    if collection.count() == 0:
        st.warning("Build an index before asking a question.")
        return

    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    history = st.session_state.messages[:-1]
    with st.chat_message("assistant"):
        try:
            with st.spinner("Retrieving grounded evidence..."):
                details = generate_answer(
                    query,
                    top_k=5,
                    history=history,
                    return_details=True,
                )
            st.markdown(details["answer"])
            sources = _sources_from_results(details["results"])
            _render_sources(sources)
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": details["answer"],
                    "sources": sources,
                }
            )
        except RuntimeError as exc:
            st.error(str(exc))


if __name__ == "__main__":
    main()
