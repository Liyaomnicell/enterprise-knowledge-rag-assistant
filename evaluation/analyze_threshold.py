import json
import statistics
from pathlib import Path


BASELINE_FILE = Path(
    "evaluation/results/baseline_v1.json"
)

JSON_OUTPUT_FILE = Path(
    "evaluation/results/threshold_analysis.json"
)

MARKDOWN_OUTPUT_FILE = Path(
    "evaluation/results/threshold_analysis.md"
)


def load_baseline():
    with open(
        BASELINE_FILE,
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


def extract_scores(baseline):
    positive_scores = [
        item["best_expected_score"]
        for item in baseline["answerable_results"]
    ]

    negative_scores = [
        item["top_score"]
        for item in baseline["unknown_results"]
    ]

    return positive_scores, negative_scores


def calculate_distribution(scores):
    if not scores:
        return {
            "count": 0,
            "min": 0.0,
            "max": 0.0,
            "mean": 0.0,
            "median": 0.0,
            "stddev": 0.0,
            "scores": [],
        }

    return {
        "count": len(scores),
        "min": round(min(scores), 6),
        "max": round(max(scores), 6),
        "mean": round(
            statistics.mean(scores),
            6,
        ),
        "median": round(
            statistics.median(scores),
            6,
        ),
        "stddev": round(
            statistics.pstdev(scores),
            6,
        ),
        "scores": sorted(scores),
    }


def generate_candidate_thresholds(
    positive_scores,
    negative_scores,
):
    all_scores = sorted(
        set(
            positive_scores
            + negative_scores
        )
    )

    if not all_scores:
        return []

    thresholds = []

    # Threshold just below the smallest score.
    thresholds.append(
        all_scores[0] - 0.001
    )

    # Midpoints between every adjacent pair of scores.
    for left, right in zip(
        all_scores,
        all_scores[1:],
    ):
        midpoint = (
            left + right
        ) / 2

        thresholds.append(
            midpoint
        )

    # Threshold just above the largest score.
    thresholds.append(
        all_scores[-1] + 0.001
    )

    return thresholds


def evaluate_threshold(
    threshold,
    positive_scores,
    negative_scores,
):
    # Positive = actually answerable.
    # Negative = actually unanswerable.

    true_positive = sum(
        score >= threshold
        for score in positive_scores
    )

    false_negative = sum(
        score < threshold
        for score in positive_scores
    )

    false_positive = sum(
        score >= threshold
        for score in negative_scores
    )

    true_negative = sum(
        score < threshold
        for score in negative_scores
    )

    total = (
        true_positive
        + false_negative
        + false_positive
        + true_negative
    )

    accuracy = (
        (
            true_positive
            + true_negative
        )
        / total
        if total > 0
        else 0.0
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

    return {
        "threshold": round(
            threshold,
            6,
        ),
        "true_positive":
            true_positive,
        "false_positive":
            false_positive,
        "true_negative":
            true_negative,
        "false_negative":
            false_negative,
        "accuracy": round(
            accuracy,
            6,
        ),
        "precision": round(
            precision,
            6,
        ),
        "recall": round(
            recall,
            6,
        ),
        "f1": round(
            f1,
            6,
        ),
    }


def select_best_threshold(
    evaluations,
):
    if not evaluations:
        return None

    return max(
        evaluations,
        key=lambda item: (
            item["f1"],
            item["precision"],
            item["accuracy"],
            item["threshold"],
        ),
    )


def save_json_report(result):
    JSON_OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        JSON_OUTPUT_FILE,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            result,
            file,
            indent=2,
            ensure_ascii=False,
        )


def save_markdown_report(result):
    positive = (
        result["positive_distribution"]
    )

    negative = (
        result["negative_distribution"]
    )

    best = (
        result["best_threshold_by_f1"]
    )

    overlap = (
        result["distributions_overlap"]
    )

    lines = []

    lines.append(
        "# Retrieval Threshold Analysis"
    )
    lines.append("")

    lines.append(
        "## Objective"
    )
    lines.append("")
    lines.append(
        "Analyze whether cosine similarity "
        "scores can distinguish answerable "
        "queries from queries that are not "
        "supported by the knowledge base."
    )
    lines.append("")

    lines.append(
        "## Positive Score Distribution"
    )
    lines.append("")
    lines.append(
        "Positive scores are the highest "
        "similarity scores from the known "
        "ground-truth documents for "
        "answerable queries."
    )
    lines.append("")
    lines.append(
        f"- Count: {positive['count']}"
    )
    lines.append(
        f"- Minimum: {positive['min']:.3f}"
    )
    lines.append(
        f"- Maximum: {positive['max']:.3f}"
    )
    lines.append(
        f"- Mean: {positive['mean']:.3f}"
    )
    lines.append(
        f"- Median: {positive['median']:.3f}"
    )
    lines.append(
        f"- Std Dev: {positive['stddev']:.3f}"
    )
    lines.append("")

    lines.append(
        "## Negative Score Distribution"
    )
    lines.append("")
    lines.append(
        "Negative scores are the Top-1 "
        "retrieval scores for queries labeled "
        "as unanswerable."
    )
    lines.append("")
    lines.append(
        f"- Count: {negative['count']}"
    )
    lines.append(
        f"- Minimum: {negative['min']:.3f}"
    )
    lines.append(
        f"- Maximum: {negative['max']:.3f}"
    )
    lines.append(
        f"- Mean: {negative['mean']:.3f}"
    )
    lines.append(
        f"- Median: {negative['median']:.3f}"
    )
    lines.append(
        f"- Std Dev: {negative['stddev']:.3f}"
    )
    lines.append("")

    lines.append(
        "## Distribution Overlap"
    )
    lines.append("")

    if overlap:
        lines.append(
            "The positive and negative score "
            "distributions overlap."
        )
        lines.append("")
        lines.append(
            "A single cosine similarity "
            "threshold cannot perfectly "
            "separate answerable and "
            "unanswerable queries on the "
            "current dataset."
        )
    else:
        lines.append(
            "The current positive and negative "
            "score distributions do not overlap."
        )
        lines.append("")
        lines.append(
            "A single similarity threshold "
            "appears viable on this dataset."
        )

    lines.append("")

    lines.append(
        "## Best Candidate Threshold"
    )
    lines.append("")

    lines.append(
        f"- Threshold: "
        f"{best['threshold']:.3f}"
    )
    lines.append(
        f"- Accuracy: "
        f"{best['accuracy']:.3f}"
    )
    lines.append(
        f"- Precision: "
        f"{best['precision']:.3f}"
    )
    lines.append(
        f"- Recall: "
        f"{best['recall']:.3f}"
    )
    lines.append(
        f"- F1: "
        f"{best['f1']:.3f}"
    )
    lines.append("")

    lines.append(
        "## Confusion Matrix"
    )
    lines.append("")
    lines.append(
        "| | Predicted Answerable | "
        "Predicted Unanswerable |"
    )
    lines.append(
        "|---|---:|---:|"
    )
    lines.append(
        f"| Actually Answerable | "
        f"{best['true_positive']} | "
        f"{best['false_negative']} |"
    )
    lines.append(
        f"| Actually Unanswerable | "
        f"{best['false_positive']} | "
        f"{best['true_negative']} |"
    )
    lines.append("")

    lines.append(
        "## Interpretation"
    )
    lines.append("")
    lines.append(
        "False positives are particularly "
        "important in RAG systems because they "
        "represent unsupported queries that "
        "would still be treated as answerable, "
        "increasing hallucination risk."
    )
    lines.append("")
    lines.append(
        "False negatives represent valid "
        "questions that the system incorrectly "
        "rejects."
    )
    lines.append("")
    lines.append(
        "The recommended threshold is only a "
        "candidate derived from the current "
        "small synthetic evaluation dataset. "
        "It should not be considered "
        "production-calibrated."
    )

    with open(
        MARKDOWN_OUTPUT_FILE,
        "w",
        encoding="utf-8",
    ) as file:
        file.write(
            "\n".join(lines)
        )


def analyze():
    baseline = load_baseline()

    (
        positive_scores,
        negative_scores,
    ) = extract_scores(
        baseline
    )

    positive_distribution = (
        calculate_distribution(
            positive_scores
        )
    )

    negative_distribution = (
        calculate_distribution(
            negative_scores
        )
    )

    thresholds = (
        generate_candidate_thresholds(
            positive_scores,
            negative_scores,
        )
    )

    evaluations = [
        evaluate_threshold(
            threshold,
            positive_scores,
            negative_scores,
        )
        for threshold in thresholds
    ]

    best_threshold = (
        select_best_threshold(
            evaluations
        )
    )

    distributions_overlap = (
        positive_distribution["min"]
        <= negative_distribution["max"]
    )

    separation_gap = (
        positive_distribution["min"]
        - negative_distribution["max"]
    )

    result = {
        "positive_distribution":
            positive_distribution,

        "negative_distribution":
            negative_distribution,

        "distributions_overlap":
            distributions_overlap,

        "separation_gap":
            round(
                separation_gap,
                6,
            ),

        "best_threshold_by_f1":
            best_threshold,

        "all_threshold_results":
            evaluations,
    }

    save_json_report(
        result
    )

    save_markdown_report(
        result
    )

    print()
    print("=" * 80)
    print("THRESHOLD ANALYSIS")
    print("=" * 80)

    print()
    print("Positive scores:")
    print(
        f"Min: "
        f"{positive_distribution['min']:.3f}"
    )
    print(
        f"Max: "
        f"{positive_distribution['max']:.3f}"
    )
    print(
        f"Mean: "
        f"{positive_distribution['mean']:.3f}"
    )

    print()
    print("Negative scores:")
    print(
        f"Min: "
        f"{negative_distribution['min']:.3f}"
    )
    print(
        f"Max: "
        f"{negative_distribution['max']:.3f}"
    )
    print(
        f"Mean: "
        f"{negative_distribution['mean']:.3f}"
    )

    print()
    print(
        f"Distributions overlap: "
        f"{distributions_overlap}"
    )

    print(
        f"Separation gap: "
        f"{separation_gap:.3f}"
    )

    print()
    print(
        "Best threshold:"
    )

    print(
        f"{best_threshold['threshold']:.3f}"
    )

    print(
        f"Precision: "
        f"{best_threshold['precision']:.3f}"
    )

    print(
        f"Recall: "
        f"{best_threshold['recall']:.3f}"
    )

    print(
        f"F1: "
        f"{best_threshold['f1']:.3f}"
    )

    print(
        f"Accuracy: "
        f"{best_threshold['accuracy']:.3f}"
    )

    print()
    print(
        f"JSON report: "
        f"{JSON_OUTPUT_FILE}"
    )

    print(
        f"Markdown report: "
        f"{MARKDOWN_OUTPUT_FILE}"
    )


if __name__ == "__main__":
    analyze()