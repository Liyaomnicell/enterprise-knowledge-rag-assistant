import json
from pathlib import Path

from evaluation.compare_chunking import (
    build_fixed_size_retriever,
    build_paragraph_retriever,
)

from evaluation.evaluate_retrieval import (
    evidence_hit_at_k,
    evidence_recall_at_k,
)


DATASET_FILE = Path(
    "data/evaluation/retrieval_eval.json"
)

CASE_ID = "retry_004"


def load_case(case_id):
    with open(
        DATASET_FILE,
        "r",
        encoding="utf-8",
    ) as file:
        dataset = json.load(file)

    for item in dataset:
        if item["id"] == case_id:
            return item

    raise ValueError(
        f"Case not found: {case_id}"
    )


def evaluate_strategy(
    name,
    retriever,
    question,
    expected_evidence,
):
    # Retrieve 5 once.
    # Then evaluate K=1,3,4,5 from the same ranking.
    results = retriever.retrieve(
        query=question,
        top_k=5,
    )

    print()
    print("=" * 80)
    print(name)
    print("=" * 80)

    print()
    print("Top-5 chunks:")

    for rank, result in enumerate(
        results,
        start=1,
    ):
        print(
            f"{rank}. "
            f"{result.chunk.document_name} / "
            f"{result.chunk.chunk_id} / "
            f"{result.score:.6f}"
        )

    print()
    print("Evidence metrics:")

    for k in [1, 3, 4, 5]:

        hit = evidence_hit_at_k(
            results,
            expected_evidence,
            k,
        )

        recall = evidence_recall_at_k(
            results,
            expected_evidence,
            k,
        )

        print(
            f"K={k}: "
            f"Hit={hit}, "
            f"Recall={recall:.3f}"
        )


def main():

    case = load_case(CASE_ID)

    question = case["question"]

    expected_evidence = (
        case["expected_evidence"]
    )

    print()
    print("=" * 80)
    print("EVIDENCE RETRIEVAL COMPARISON")
    print("=" * 80)

    print()
    print(f"Case: {CASE_ID}")
    print(f"Question: {question}")
    print(
        f"Expected evidence: "
        f"{expected_evidence}"
    )

    fixed_retriever = (
        build_fixed_size_retriever()
    )

    paragraph_retriever = (
        build_paragraph_retriever()
    )

    evaluate_strategy(
        "FIXED-SIZE CHUNKING",
        fixed_retriever,
        question,
        expected_evidence,
    )

    evaluate_strategy(
        "PARAGRAPH-AWARE CHUNKING",
        paragraph_retriever,
        question,
        expected_evidence,
    )


if __name__ == "__main__":
    main()