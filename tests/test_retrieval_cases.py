from evaluation.compare_chunking import (
    build_fixed_size_retriever,
)

from evaluation.evaluate_retrieval import (
    load_evaluation_dataset,
)

from evaluation.config.quality_gates import (
    RETRIEVAL_CASE_GATES,
)

from evaluation.config.quality_gates import (
    RETRIEVAL_CASE_GATES,
    MULTI_DOCUMENT_CASE_GATES,
)

def find_case(dataset, case_id):
    for case in dataset:
        if case["id"] == case_id:
            return case

    raise ValueError(
        f"Evaluation case not found: {case_id}"
    )


def get_ranked_documents(results):
    """
    Convert chunk-level retrieval results into
    a deduplicated document-level ranking.

    Example:

    chunks:
        api_timeout-1
        api_timeout-0
        service_retry_policy-1

    becomes:

        [
            "api_timeout.md",
            "service_retry_policy.md"
        ]
    """

    documents = []

    for result in results:
        document_name = (
            result.chunk.document_name
        )

        if document_name not in documents:
            documents.append(
                document_name
            )

    return documents


def test_anchor_retrieval_cases():
    dataset = load_evaluation_dataset()

    retriever = (
        build_fixed_size_retriever()
    )

    for case_id, gate in (
        RETRIEVAL_CASE_GATES.items()
    ):
        case = find_case(
            dataset,
            case_id,
        )

        top_k = gate["top_k"]

        # Retrieve enough chunks first because
        # multiple chunks may come from the same
        # document.
        results = retriever.retrieve(
            query=case["question"],
            top_k=10,
        )

        ranked_documents = (
            get_ranked_documents(
                results
            )
        )

        top_documents = (
            ranked_documents[:top_k]
        )

        expected_documents = set(
            case["expected_documents"]
        )

        retrieved_documents = set(
            top_documents
        )

        missing_documents = (
            expected_documents
            - retrieved_documents
        )

        assert not missing_documents, (
            f"\nRetrieval regression for "
            f"{case_id}\n"
            f"Question: "
            f"{case['question']}\n"
            f"Expected documents: "
            f"{sorted(expected_documents)}\n"
            f"Top-{top_k} documents: "
            f"{top_documents}\n"
            f"Missing: "
            f"{sorted(missing_documents)}"
        )

def test_multi_document_retrieval_cases():
    dataset = load_evaluation_dataset()

    retriever = (
        build_fixed_size_retriever()
    )

    for case_id, gate in (
        MULTI_DOCUMENT_CASE_GATES.items()
    ):
        case = find_case(
            dataset,
            case_id,
        )

        top_k = gate["top_k"]
        min_recall = gate["min_recall"]

        # Retrieve more chunks than documents,
        # because several chunks can belong
        # to the same document.
        results = retriever.retrieve(
            query=case["question"],
            top_k=10,
        )

        documents = get_ranked_documents(
            results
        )

        top_documents = documents[:top_k]

        expected_documents = set(
            case["expected_documents"]
        )

        retrieved_documents = set(
            top_documents
        )

        matched_documents = (
            expected_documents
            & retrieved_documents
        )

        recall = (
            len(matched_documents)
            / len(expected_documents)
        )

        assert recall >= min_recall, (
            f"\nMulti-document retrieval "
            f"regression for {case_id}\n"
            f"Question: "
            f"{case['question']}\n"
            f"Expected documents: "
            f"{sorted(expected_documents)}\n"
            f"Top-{top_k} documents: "
            f"{top_documents}\n"
            f"Recall@{top_k}: "
            f"{recall:.3f}\n"
            f"Required: "
            f"{min_recall:.3f}"
        )