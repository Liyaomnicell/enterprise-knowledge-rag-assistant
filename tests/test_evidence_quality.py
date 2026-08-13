import pytest

from evaluation.evaluate_evidence_sufficiency import (
    run_evaluation,
)

from evaluation.config.quality_gates import (
    EVIDENCE_SUFFICIENCY_QUALITY_GATES,
)


@pytest.mark.llm
def test_evidence_sufficiency_quality():
    output = run_evaluation()

    metrics = output["summary"]

    gates = (
        EVIDENCE_SUFFICIENCY_QUALITY_GATES
    )

    assert (
        metrics["false_positive"]
        <= gates["max_false_positives"]
    ), (
        f"False-positive regression: "
        f"{metrics['false_positive']} > "
        f"{gates['max_false_positives']}"
    )

    assert (
        metrics["precision"]
        >= gates["min_precision"]
    ), (
        f"Precision regression: "
        f"{metrics['precision']:.3f} < "
        f"{gates['min_precision']:.3f}"
    )

    assert (
        metrics["accuracy"]
        >= gates["min_accuracy"]
    ), (
        f"Accuracy regression: "
        f"{metrics['accuracy']:.3f} < "
        f"{gates['min_accuracy']:.3f}"
    )

    assert (
        metrics["recall"]
        >= gates["min_recall"]
    ), (
        f"Recall regression: "
        f"{metrics['recall']:.3f} < "
        f"{gates['min_recall']:.3f}"
    )

    assert (
        metrics["f1"]
        >= gates["min_f1"]
    ), (
        f"F1 regression: "
        f"{metrics['f1']:.3f} < "
        f"{gates['min_f1']:.3f}"
    )