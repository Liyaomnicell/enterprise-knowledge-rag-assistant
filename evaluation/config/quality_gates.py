RETRIEVAL_QUALITY_GATES = {
    "hit_at_1": 0.95,
    "hit_at_3": 1.00,
    "mrr": 0.97,
    "recall_at_1": 0.90,
    "recall_at_3": 1.00,
}


EVIDENCE_SUFFICIENCY_QUALITY_GATES = {
    "accuracy": 0.95,
    "precision": 1.00,
    "recall": 0.95,
    "f1": 0.97,
    "max_false_positives": 0,
}

RETRIEVAL_CASE_GATES = {
    "retry_004": {
        "top_k": 3,
    },
    "cache_004": {
        "top_k": 3,
    },
    "api_004": {
        "top_k": 3,
    },
}

MULTI_DOCUMENT_CASE_GATES = {
    "multi_001": {
        "top_k": 3,
        "min_recall": 1.0,
    },
    "multi_002": {
        "top_k": 3,
        "min_recall": 1.0,
    },
}

EVIDENCE_SUFFICIENCY_QUALITY_GATES = {
    "min_accuracy": 0.93,
    "min_precision": 1.00,
    "min_recall": 0.90,
    "min_f1": 0.95,
    "max_false_positives": 0,
}