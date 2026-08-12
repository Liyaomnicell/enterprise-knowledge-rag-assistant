from evaluation.evaluate_retrieval import (
    load_evaluation_dataset,
    build_retriever,
)


def analyze():

    dataset = (
        load_evaluation_dataset()
    )

    retriever = build_retriever()

    print()
    print("=" * 80)
    print("UNKNOWN QUERY ANALYSIS")
    print("=" * 80)

    for item in dataset:

        if item["answerable"]:
            continue

        results = retriever.retrieve(
            query=item["question"],
            top_k=3,
        )

        print()
        print("-" * 80)

        print(
            f"Question: "
            f"{item['question']}"
        )

        for result in results:

            print(
                f"{result.score:.4f} "
                f"{result.chunk.document_name}"
            )


if __name__ == "__main__":
    analyze()