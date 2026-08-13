import json
from pathlib import Path


BASELINE_FILE = Path(
    "evaluation/results/baseline_v1.json"
)


def load_baseline():
    with open(
        BASELINE_FILE,
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


def print_hit_at_1_failures(
    answerable_results,
):
    print()
    print("=" * 80)
    print("HIT@1 FAILURES")
    print("=" * 80)

    failures = [
        item
        for item in answerable_results
        if not item["hit_at_1"]
    ]

    if not failures:
        print()
        print("No Hit@1 failures.")
        return

    for item in failures:
        print()
        print("-" * 80)

        print(
            f"ID: {item['id']}"
        )

        print(
            f"Category: {item['category']}"
        )

        print(
            f"Question: {item['question']}"
        )

        print(
            "Expected documents:"
        )

        for document in item[
            "expected_documents"
        ]:
            print(
                f"  - {document}"
            )

        print(
            "Top 3 document ranking:"
        )

        for index, document in enumerate(
            item["ranked_documents"][:3],
            start=1,
        ):
            print(
                f"  {index}. {document}"
            )

        print(
            f"Recall@1: "
            f"{item['recall_at_1']:.3f}"
        )

        print(
            f"Recall@3: "
            f"{item['recall_at_3']:.3f}"
        )

        print(
            f"Best expected score: "
            f"{item['best_expected_score']:.3f}"
        )

        print()
        print("Top retrieved chunks:")

        for result in item[
            "retrieved_chunks"
        ][:5]:

            print(
                f"  {result['score']:.3f}  "
                f"{result['document_name']}  "
                f"{result['chunk_id']}"
            )


def print_low_recall_at_1_cases(
    answerable_results,
):
    print()
    print("=" * 80)
    print("RECALL@1 < 1.0")
    print("=" * 80)

    cases = [
        item
        for item in answerable_results
        if item["recall_at_1"] < 1.0
    ]

    if not cases:
        print()
        print(
            "No Recall@1 failures."
        )
        return

    for item in cases:
        print()
        print("-" * 80)

        print(
            f"ID: {item['id']}"
        )

        print(
            f"Question: "
            f"{item['question']}"
        )

        print(
            f"Expected: "
            f"{item['expected_documents']}"
        )

        print(
            f"Top 3: "
            f"{item['ranked_documents'][:3]}"
        )

        print(
            f"Recall@1: "
            f"{item['recall_at_1']:.3f}"
        )

        print(
            f"Recall@3: "
            f"{item['recall_at_3']:.3f}"
        )


def print_high_scoring_negatives(
    unknown_results,
    minimum_score=0.30,
):
    print()
    print("=" * 80)
    print(
        "HIGH-SCORING UNANSWERABLE QUERIES"
    )
    print("=" * 80)

    cases = [
        item
        for item in unknown_results
        if item["top_score"]
        >= minimum_score
    ]

    cases.sort(
        key=lambda item:
            item["top_score"],
        reverse=True,
    )

    if not cases:
        print()
        print(
            "No high-scoring negative queries."
        )
        return

    for item in cases:
        print()
        print("-" * 80)

        print(
            f"ID: {item['id']}"
        )

        print(
            f"Category: "
            f"{item['category']}"
        )

        print(
            f"Question: "
            f"{item['question']}"
        )

        print(
            f"Top score: "
            f"{item['top_score']:.3f}"
        )

        print()
        print(
            "Top retrieved chunks:"
        )

        for result in item[
            "retrieved_chunks"
        ]:

            print(
                f"  {result['score']:.3f}  "
                f"{result['document_name']}  "
                f"{result['chunk_id']}"
            )


def analyze():
    baseline = load_baseline()

    answerable_results = (
        baseline[
            "answerable_results"
        ]
    )

    unknown_results = (
        baseline[
            "unknown_results"
        ]
    )

    print_hit_at_1_failures(
        answerable_results
    )

    print_low_recall_at_1_cases(
        answerable_results
    )

    print_high_scoring_negatives(
        unknown_results,
        minimum_score=0.30,
    )


if __name__ == "__main__":
    analyze()