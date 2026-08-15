import json
from pathlib import Path

from app.rag.document_loader import (
    load_documents,
)
from app.rag.chunker import (
    chunk_documents,
)
from app.rag.embedding import (
    EmbeddingService,
)
from app.rag.in_memory_vector_store import (
    InMemoryVectorStore,
)
from app.rag.indexing import (
    IndexingService,
)
from app.rag.retriever import (
    SemanticRetriever,
)
from app.rag.evidence_checker import (
    EvidenceSufficiency,
    EvidenceSufficiencyChecker,
)


EVALUATION_FILE = Path(
    "data/evaluation/retrieval_eval.json"
)

OUTPUT_FILE = Path(
    "evaluation/results/"
    "evidence_sufficiency_v1.json"
)


def load_dataset():
    with open(
        EVALUATION_FILE,
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


def build_retriever():
    documents = load_documents(
        "data/documents"
    )

    chunks = chunk_documents(
        documents,
        chunk_size=500,
        chunk_overlap=100,
    )

    embedding_service = (
        EmbeddingService()
    )

    vector_store = (
        InMemoryVectorStore()
    )

    indexing_service = (
        IndexingService(
            embedding_service=embedding_service,
            vector_store=vector_store,
        )
    )

    indexing_service.index(
        chunks
    )

    return SemanticRetriever(
        embedding_service=embedding_service,
        vector_store=vector_store,
    )


def run_evaluation():
    """
    Run the evidence sufficiency evaluation
    and return the calculated metrics and
    per-question results.

    This function does not write files and
    does not print the final summary.

    It can therefore be reused by pytest.
    """

    dataset = load_dataset()

    retriever = build_retriever()

    checker = (
        EvidenceSufficiencyChecker()
    )

    results = []

    true_positive = 0
    false_positive = 0
    true_negative = 0
    false_negative = 0

    for item in dataset:

        question = item["question"]

        retrieval_results = (
            retriever.retrieve(
                query=question,
                top_k=3,
            )
        )

        prediction = checker.check(
            question=question,
            retrieval_results=retrieval_results,
        )

        predicted_answerable = (
            prediction
            == EvidenceSufficiency.SUFFICIENT
        )

        actual_answerable = (
            item["answerable"]
        )

        if (
            actual_answerable
            and predicted_answerable
        ):
            outcome = "TP"
            true_positive += 1

        elif (
            not actual_answerable
            and predicted_answerable
        ):
            outcome = "FP"
            false_positive += 1

        elif (
            not actual_answerable
            and not predicted_answerable
        ):
            outcome = "TN"
            true_negative += 1

        else:
            outcome = "FN"
            false_negative += 1

        results.append(
            {
                "id":
                    item["id"],

                "question":
                    question,

                "actual_answerable":
                    actual_answerable,

                "prediction":
                    prediction.value,

                "outcome":
                    outcome,

                "retrieved_documents":
                    [
                        {
                            "document_name":
                                result.chunk.document_name,

                            "chunk_id":
                                result.chunk.chunk_id,

                            "score":
                                round(
                                    result.score,
                                    6,
                                ),
                        }
                        for result
                        in retrieval_results
                    ],
            }
        )

    total = len(dataset)

    accuracy = (
        (
            true_positive
            + true_negative
        )
        / total
    )

    precision = (
        true_positive
        / (
            true_positive
            + false_positive
        )
        if (
            true_positive
            + false_positive
        ) > 0
        else 0.0
    )

    recall = (
        true_positive
        / (
            true_positive
            + false_negative
        )
        if (
            true_positive
            + false_negative
        ) > 0
        else 0.0
    )

    f1 = (
        2
        * precision
        * recall
        / (
            precision
            + recall
        )
        if (
            precision
            + recall
        ) > 0
        else 0.0
    )

    summary = {
        "question_count":
            total,

        "true_positive":
            true_positive,

        "false_positive":
            false_positive,

        "true_negative":
            true_negative,

        "false_negative":
            false_negative,

        "accuracy":
            round(
                accuracy,
                6,
            ),

        "precision":
            round(
                precision,
                6,
            ),

        "recall":
            round(
                recall,
                6,
            ),

        "f1":
            round(
                f1,
                6,
            ),
    }

    return {
        "summary": summary,
        "results": results,
    }


def save_result(output):
    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            output,
            file,
            indent=2,
            ensure_ascii=False,
        )


def print_result(output):
    summary = output["summary"]

    print()
    print("=" * 80)

    print(
        "EVIDENCE SUFFICIENCY EVALUATION"
    )

    print(
        "=" * 80
    )

    print(
        f"Questions: "
        f"{summary['question_count']}"
    )

    print(
        f"TP: "
        f"{summary['true_positive']}"
    )

    print(
        f"FP: "
        f"{summary['false_positive']}"
    )

    print(
        f"TN: "
        f"{summary['true_negative']}"
    )

    print(
        f"FN: "
        f"{summary['false_negative']}"
    )

    print()

    print(
        f"Accuracy: "
        f"{summary['accuracy']:.3f}"
    )

    print(
        f"Precision: "
        f"{summary['precision']:.3f}"
    )

    print(
        f"Recall: "
        f"{summary['recall']:.3f}"
    )

    print(
        f"F1: "
        f"{summary['f1']:.3f}"
    )

    print()

    print(
        f"Result: "
        f"{OUTPUT_FILE}"
    )


def main():
    output = run_evaluation()

    save_result(output)

    print_result(output)


if __name__ == "__main__":
    main()