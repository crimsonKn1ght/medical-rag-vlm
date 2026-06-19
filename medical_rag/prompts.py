from typing import Iterable, Tuple

from .retrieval import RetrievalResult


def build_context(results: Iterable[RetrievalResult], token_budget: int = 900, include_scores: bool = False) -> Tuple[str, int]:
    chunks = []
    used_tokens = 0
    for result in results:
        heading = f"Retrieved case {result.rank}: {result.case_id}"
        if include_scores:
            heading += f" (score={result.score:.4f}, source={result.source})"
        evidence = []
        if result.findings:
            evidence.append(f"Findings: {result.findings}")
        if result.impression:
            evidence.append(f"Impression: {result.impression}")
        if not evidence:
            evidence.append(f"Report: {result.report_text}")
        block = heading + "\n" + "\n".join(evidence)
        block_tokens = len(block.split())
        if chunks and used_tokens + block_tokens > token_budget:
            break
        if not chunks and block_tokens > token_budget:
            words = block.split()[:token_budget]
            block = " ".join(words)
            block_tokens = len(words)
        chunks.append(block)
        used_tokens += block_tokens
    return "\n\n".join(chunks), used_tokens


def build_medical_vqa_prompt(question: str, context: str = "") -> str:
    instruction = (
        "Answer the clinical visual question using the target image. "
        "If retrieved evidence is provided, use it only as grounding context. "
        "Do not invent findings that are unsupported by the image or evidence."
    )
    if context:
        return f"{instruction}\n\nRetrieved evidence:\n{context}\n\nQuestion: {question}"
    return f"{instruction}\n\nQuestion: {question}"
