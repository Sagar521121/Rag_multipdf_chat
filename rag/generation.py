"""Grounded Gemini generation after retrieval and refusal checks."""

from __future__ import annotations

import os
from collections.abc import Sequence

import streamlit as st
from dotenv import load_dotenv
from google import genai

from rag.augmentation import build_context
from rag.relevance import REFUSAL, has_relevant_results
from rag.retrieval import retrieve


load_dotenv()

MODEL_NAME = os.getenv(
    "GEMINI_MODEL",
    "gemini-3.6-flash",
)


@st.cache_resource(show_spinner=False)
def get_gemini_client() -> genai.Client:
    """Create and cache the Gemini client."""

    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY is not configured"
        )

    return genai.Client(api_key=api_key)


def _history_text(
    history: Sequence[dict] | None,
) -> str:
    """Format recent conversation history for the prompt."""

    if not history:
        return ""

    recent = history[-6:]

    return "\n".join(
        f"{message.get('role', 'user').title()}: "
        f"{str(message.get('content', ''))[:500]}"
        for message in recent
    )


def _retrieval_query(
    query: str,
    history: Sequence[dict] | None,
) -> str:
    """
    Build a retrieval query without turning previous answers into evidence.

    Ordinary new questions are embedded by themselves.

    For follow-up questions, recent user questions are included so that
    references such as "its", "the study", or "the paper" can be resolved
    without injecting previous assistant answers into semantic retrieval.
    """

    if not history:
        return query

    normalized = query.lower()

    follow_up_markers = (
        " it ",
        " its ",
        " they ",
        " them ",
        " this ",
        " that ",
        " these ",
        " those ",
        " he ",
        " she ",
        " the study ",
        " the paper ",
        " the device ",
    )

    padded_query = f" {normalized.strip()} "

    if not any(
        marker in padded_query
        for marker in follow_up_markers
    ):
        return query

    recent_user_turns = [
        str(message.get("content", ""))[:500]
        for message in history[-6:]
        if message.get("role") == "user"
        and message.get("content")
    ]

    if not recent_user_turns:
        return query

    return (
        "Previous user questions:\n"
        + "\n".join(recent_user_turns)
        + f"\nCurrent question:\n{query}"
    )


def _prompt(
    query: str,
    context: str,
    history: Sequence[dict] | None,
) -> str:
    """Build the grounded Gemini prompt."""

    history_text = _history_text(history)

    return f"""You are a concise research-paper assistant.

Use ONLY the retrieved document context below for factual claims. The
conversation is only for resolving follow-up references; it is not evidence.
Do not use outside knowledge, invent details, or repeat whole passages.
Answer directly in a few sentences. Cite factual claims as [source, Page X].

Recent conversation:
{history_text or "(none)"}

Retrieved context:
{context}

Current question:
{query}

Answer:
"""


def generate_answer(
    query: str,
    top_k: int = 5,
    history: Sequence[dict] | None = None,
    return_details: bool = False,
):
    """Retrieve, refuse when unsupported, then generate a grounded answer."""

    # --------------------------------------------------------
    # Build retrieval query
    # --------------------------------------------------------

    retrieval_query = _retrieval_query(
        query,
        history,
    )

    # --------------------------------------------------------
    # Retrieve a larger candidate pool
    # --------------------------------------------------------

    candidate_k = max(top_k * 3, 12)

    results = retrieve(
        retrieval_query,
        top_k=candidate_k,
    )

    # --------------------------------------------------------
    # Relevance check
    # --------------------------------------------------------

    relevant = has_relevant_results(
        results,
        query=retrieval_query,
    )

    # --------------------------------------------------------
    # Build context
    # --------------------------------------------------------

    context = build_context(
        results,
        max_chunks=top_k,
    )

    # --------------------------------------------------------
    # Refuse or generate
    # --------------------------------------------------------

    if not relevant:
        answer = REFUSAL

    else:
        try:
            response = get_gemini_client().models.generate_content(
                model=MODEL_NAME,
                contents=_prompt(
                    query,
                    context,
                    history,
                ),
            )

            answer = (
                response.text or ""
            ).strip()

            if not answer:
                raise RuntimeError(
                    "Gemini returned an empty response"
                )

        except Exception as exc:
            raise RuntimeError(
                f"Generation failed: {exc}"
            ) from exc

    # --------------------------------------------------------
    # Return details
    # --------------------------------------------------------

    if return_details:
        return {
            "answer": answer,
            "results": results,
            "context": context,
            "relevant": relevant,
            "retrieval_query": retrieval_query,
        }

    return answer
