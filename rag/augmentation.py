"""Concise, source-attributed context construction."""

from __future__ import annotations


def build_context(results: dict, max_chunks: int = 5) -> str:
    documents = (results.get("documents") or [[]])[0]
    metadatas = (results.get("metadatas") or [[]])[0]
    distances = (results.get("distances") or [[]])[0]
    context_parts: list[str] = []
    seen: set[tuple[str, int, int]] = set()

    for index, (document, metadata) in enumerate(zip(documents, metadatas)):
        metadata = metadata or {}
        source = str(metadata.get("source", "unknown"))
        page = int(metadata.get("page", 0))
        chunk_id = int(metadata.get("chunk_id", 0))
        identity = (source, page, chunk_id)
        if identity in seen:
            continue
        seen.add(identity)
        distance = distances[index] if index < len(distances) else None
        distance_label = f" | Distance: {float(distance):.3f}" if distance is not None else ""
        context_parts.append(
            f"[Source: {source} | Page: {page} | Chunk: {chunk_id}{distance_label}]\n"
            f"{str(document).strip()}"
        )
        if len(context_parts) >= max_chunks:
            break

    return "\n\n---\n\n".join(context_parts)
