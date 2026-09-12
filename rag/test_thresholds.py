from rag.retrieval import retrieve


queries = [
    ("What is BERT?", True),
    ("What are BERT's pre-training tasks?", True),
    ("What is the Transformer architecture?", True),
    ("What is the main contribution of this paper?", True),

    ("What is the capital of France?", False),
    ("Who is the Prime Minister of India?", False),
    ("What is GPT-3?", False),
]


for query, expected_relevant in queries:

    results = retrieve(
        query,
        top_k=5
    )

    distances = results.get(
        "distances",
        [[]]
    )[0]

    best_distance = (
        distances[0]
        if distances
        else None
    )

    print(
        f"{best_distance:.4f} | "
        f"Expected: {expected_relevant} | "
        f"{query}"
    )