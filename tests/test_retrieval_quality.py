from evaluation.compare_chunking import (
    build_fixed_size_retriever,
)

from evaluation.evaluate_retrieval import (
    load_evaluation_dataset,
    evaluate_answerable_queries,
)

from evaluation.config.quality_gates import (
    RETRIEVAL_QUALITY_GATES,
)


def test_retrieval_quality_gates():

    dataset = load_evaluation_dataset()

    retriever = (
        build_fixed_size_retriever()
    )

    metrics, _ = (
        evaluate_answerable_queries(
            dataset,
            retriever,
        )
    )

    gates = (
        RETRIEVAL_QUALITY_GATES
    )

    assert (
        metrics["hit_at_1"]
        >= gates["hit_at_1"]
    ), (
        f"Hit@1 regression: "
        f"{metrics['hit_at_1']:.3f} "
        f"< {gates['hit_at_1']:.3f}"
    )

    assert (
        metrics["hit_at_3"]
        >= gates["hit_at_3"]
    ), (
        f"Hit@3 regression: "
        f"{metrics['hit_at_3']:.3f} "
        f"< {gates['hit_at_3']:.3f}"
    )

    assert (
        metrics["mrr"]
        >= gates["mrr"]
    ), (
        f"MRR regression: "
        f"{metrics['mrr']:.3f} "
        f"< {gates['mrr']:.3f}"
    )

    assert (
        metrics["recall_at_1"]
        >= gates["recall_at_1"]
    ), (
        f"Recall@1 regression: "
        f"{metrics['recall_at_1']:.3f} "
        f"< {gates['recall_at_1']:.3f}"
    )

    assert (
        metrics["recall_at_3"]
        >= gates["recall_at_3"]
    ), (
        f"Recall@3 regression: "
        f"{metrics['recall_at_3']:.3f} "
        f"< {gates['recall_at_3']:.3f}"
    )