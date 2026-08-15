from evaluation.evaluate_retrieval import (
    build_retriever,
)


def main():
    question = (
        "What are common risks "
        "of application caching?"
    )

    for store_type in [
        "in_memory",
        "faiss_flat",
        "faiss_hnsw",
    ]:
        retriever = build_retriever(
            vector_store_type=store_type
        )

        results = retriever.retrieve(
            query=question,
            top_k=3,
        )

        print()
        print("=" * 80)
        print(f"VECTOR STORE: {store_type}")
        print("=" * 80)

        for rank, result in enumerate(
            results,
            start=1,
        ):
            print(
                f"{rank}. "
                f"{result.chunk.chunk_id} "
                f"{result.score:.6f}"
            )


if __name__ == "__main__":
    main()