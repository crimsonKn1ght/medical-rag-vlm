import re
from collections import Counter
from typing import Dict, Iterable, List


def normalize_answer(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def exact_match(prediction: str, reference: str) -> float:
    return float(normalize_answer(prediction) == normalize_answer(reference))


class NLIFactualConsistencyScorer:
    """NLI-style scorer with a deterministic lexical fallback.

    The fallback keeps the pipeline testable without downloading an NLI model. If a
    transformers zero-shot or NLI model is configured later, this class can be
    replaced behind the same score() contract.
    """

    def score(self, premise: str, hypothesis: str) -> Dict[str, object]:
        premise_tokens = Counter(normalize_answer(premise).split())
        hypo_tokens = normalize_answer(hypothesis).split()
        if not hypo_tokens:
            label = "unsupported"
            confidence = 0.0
        else:
            supported = sum(1 for token in hypo_tokens if premise_tokens[token] > 0)
            ratio = supported / len(hypo_tokens)
            if ratio >= 0.8:
                label = "entailed"
            elif any(term in normalize_answer(hypothesis) for term in ["not", "no"]) and ratio < 0.4:
                label = "contradiction"
            else:
                label = "unsupported"
            confidence = ratio
        return {"label": label, "confidence": float(confidence)}


def hallucination_category(prediction: str, reference: str, nli_label: str) -> str:
    if not prediction.strip():
        return "answer_abstention"
    if exact_match(prediction, reference):
        return "supported_correct"
    if nli_label == "contradiction":
        return "contradiction"
    if nli_label == "unsupported":
        return "unsupported_assertion"
    return "answer_mismatch"


def aggregate_metrics(rows: Iterable[Dict[str, object]]) -> Dict[str, object]:
    materialized: List[Dict[str, object]] = list(rows)
    total = len(materialized)
    if total == 0:
        return {"count": 0, "exact_match": 0.0, "factual_consistency": 0.0, "categories": {}}
    categories = Counter(str(row.get("hallucination_category", "unknown")) for row in materialized)
    return {
        "count": total,
        "exact_match": sum(float(row.get("exact_match", 0.0)) for row in materialized) / total,
        "factual_consistency": sum(float(row.get("nli_confidence", 0.0)) for row in materialized) / total,
        "categories": dict(categories),
    }
