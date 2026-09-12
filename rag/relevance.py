"""Explainable retrieval acceptance and refusal checks."""

from __future__ import annotations

import os
import re


REFUSAL = "I don't have enough information in the uploaded documents."


def configured_distance_threshold() -> float:
    """
    Read the maximum accepted Chroma distance.

    This is a configurable baseline, not a universal cutoff.
    """

    raw = os.getenv(
        "RAG_DISTANCE_THRESHOLD",
        "1.10",
    )

    try:
        value = float(raw)
    except ValueError as exc:
        raise ValueError(
            "RAG_DISTANCE_THRESHOLD must be a number"
        ) from exc

    if value <= 0:
        raise ValueError(
            "RAG_DISTANCE_THRESHOLD must be greater than zero"
        )

    return value


def _normalise(text: str) -> str:
    """
    Normalize text for simple phrase matching.
    """

    return re.sub(
        r"[^a-z0-9]+",
        " ",
        text.lower(),
    ).strip()


def _specific_phrase_requirements(
    query: str,
) -> list[str]:
    """
    Detect version/entity phrases such as:

    GPT-3
    BERT-Base
    Model 2.1
    """

    normalised = _normalise(query)

    requirements: list[str] = []

    for match in re.findall(
        r"\b[a-z]+\s*[- ]\s*\d+(?:\.\d+)*\b",
        normalised,
    ):

        requirements.append(
            re.sub(
                r"\s+",
                " ",
                match.replace("-", " "),
            )
        )

    return requirements


def has_relevant_results(
    results: dict,
    query: str | None = None,
    threshold: float | None = None,
) -> bool:
    """
    Decide whether retrieved evidence is sufficiently relevant.

    Current policy:

    1. There must be at least one retrieved result.
    2. The best result must be within the configured distance.
    3. Versioned/specific entities in the query must appear
       somewhere in the retrieved context.

    This is intentionally simple and explainable.
    """

    distances = results.get(
        "distances",
        [[]],
    )

    documents = results.get(
        "documents",
        [[]],
    )

    if not distances or not distances[0]:
        return False

    best_distance = float(
        distances[0][0]
    )

    cutoff = (
        configured_distance_threshold()
        if threshold is None
        else float(threshold)
    )

    # --------------------------------------------------------
    # Distance check
    # --------------------------------------------------------

    if best_distance > cutoff:
        return False

    # --------------------------------------------------------
    # Specific entity check
    # --------------------------------------------------------

    if query:

        retrieved_documents = (
            documents[0]
            if documents
            else []
        )

        context = _normalise(
            " ".join(retrieved_documents)
        )

        for requirement in _specific_phrase_requirements(
            query
        ):

            if requirement not in context:
                return False

    return True