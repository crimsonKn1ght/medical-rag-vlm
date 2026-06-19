import argparse
import csv
import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List

from .backends import build_backend
from .data import VQASample, load_mimic_cxr_corpus, load_vqa_dataset
from .metrics import NLIFactualConsistencyScorer, aggregate_metrics, exact_match, hallucination_category
from .prompts import build_context
from .retrieval import (
    BM25Index,
    DenseImageIndex,
    LexicalOverlapReranker,
    build_visual_index,
    hybrid_search,
    save_results_jsonl,
)

logger = logging.getLogger(__name__)


def load_config(path: str) -> Dict[str, Any]:
    import yaml

    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def timestamped_output_dir(config: Dict[str, Any], command: str) -> Path:
    root = Path(config.get("outputs", {}).get("root_dir", "outputs"))
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    target = root / f"{command}-{stamp}"
    target.mkdir(parents=True, exist_ok=True)
    return target


def load_vqa_splits(config: Dict[str, Any]) -> List[VQASample]:
    datasets = config.get("datasets", {}).get("vqa", [])
    samples: List[VQASample] = []
    for dataset_cfg in datasets:
        samples.extend(
            load_vqa_dataset(
                path=dataset_cfg["path"],
                image_root=dataset_cfg.get("image_root"),
                dataset=dataset_cfg.get("name", ""),
            )
        )
    return samples


def run_baseline_eval(config: Dict[str, Any]) -> Path:
    backend = build_backend(config["model"])
    samples = load_vqa_splits(config)
    output_dir = timestamped_output_dir(config, "baseline-eval")
    rows = _evaluate_samples(backend, samples, context_lookup=None)
    _write_eval_outputs(output_dir, rows)
    return output_dir


def run_build_index(config: Dict[str, Any]) -> Path:
    backend = build_backend(config["model"])
    corpus_cfg = config["datasets"]["mimic_cxr"]
    index_dir = Path(config.get("retrieval", {}).get("index_dir", "outputs/indexes/mimic_cxr"))
    index_dir.mkdir(parents=True, exist_ok=True)

    cases = load_mimic_cxr_corpus(
        manifest_path=corpus_cfg["manifest_path"],
        image_root=corpus_cfg.get("image_root"),
        report_root=corpus_cfg.get("report_root"),
    )
    logger.info("Loaded %d corpus cases", len(cases))

    bm25 = BM25Index(cases)
    bm25.save(str(index_dir / "bm25_index.json"))

    visual = build_visual_index(cases, backend)
    visual.save(str(index_dir))

    manifest = {
        "case_count": len(cases),
        "bm25_index": str(index_dir / "bm25_index.json"),
        "visual_index_dir": str(index_dir),
        "created_at": datetime.now().isoformat(),
    }
    (index_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return index_dir


def run_rag_eval(config: Dict[str, Any], mode: str) -> Path:
    backend = build_backend(config["model"])
    samples = load_vqa_splits(config)
    retrieval_cfg = config.get("retrieval", {})
    index_dir = Path(retrieval_cfg.get("index_dir", "outputs/indexes/mimic_cxr"))

    bm25 = BM25Index.load(str(index_dir / "bm25_index.json")) if mode in {"bm25", "hybrid"} else None
    visual = DenseImageIndex.load(str(index_dir)) if mode in {"visual", "hybrid"} else None
    reranker = LexicalOverlapReranker() if retrieval_cfg.get("rerank", True) and mode == "hybrid" else None

    top_k = int(retrieval_cfg.get("top_k", 3))
    candidate_pool_size = int(retrieval_cfg.get("candidate_pool_size", 10))
    visual_weight = float(retrieval_cfg.get("visual_weight", 0.5))
    text_weight = float(retrieval_cfg.get("text_weight", 0.5))
    token_budget = int(config.get("prompting", {}).get("context_token_budget", 900))
    include_scores = bool(config.get("prompting", {}).get("include_scores_in_context", False))

    context_lookup: Dict[str, Dict[str, Any]] = {}
    for sample in samples:
        query_vector = backend.encode_image(sample.image_path, mode="global")
        results = hybrid_search(
            query=sample.question,
            query_vector=query_vector,
            bm25_index=bm25,
            visual_index=visual,
            top_k=top_k,
            candidate_pool_size=candidate_pool_size,
            visual_weight=visual_weight,
            text_weight=text_weight,
            reranker=reranker,
        )
        context, context_tokens = build_context(results, token_budget=token_budget, include_scores=include_scores)
        context_lookup[sample.sample_id] = {
            "context": context,
            "context_tokens": context_tokens,
            "retrieval": [result.to_dict() for result in results],
        }

    output_dir = timestamped_output_dir(config, f"rag-eval-{mode}")
    rows = _evaluate_samples(backend, samples, context_lookup=context_lookup)
    _write_eval_outputs(output_dir, rows)
    return output_dir


def _evaluate_samples(backend, samples: Iterable[VQASample], context_lookup=None) -> List[Dict[str, Any]]:
    scorer = NLIFactualConsistencyScorer()
    rows: List[Dict[str, Any]] = []
    for sample in samples:
        extra = context_lookup.get(sample.sample_id, {}) if context_lookup else {}
        context = extra.get("context")
        start = time.time()
        prediction = backend.generate(sample.image_path, sample.question, context=context)
        latency_ms = (time.time() - start) * 1000
        premise = "\n".join(x for x in [context or "", sample.answer] if x)
        nli = scorer.score(premise=premise, hypothesis=prediction)
        em = exact_match(prediction, sample.answer)
        rows.append(
            {
                "sample_id": sample.sample_id,
                "dataset": sample.dataset,
                "image_path": sample.image_path,
                "question": sample.question,
                "reference_answer": sample.answer,
                "prediction": prediction,
                "exact_match": em,
                "nli_label": nli["label"],
                "nli_confidence": nli["confidence"],
                "hallucination_category": hallucination_category(prediction, sample.answer, str(nli["label"])),
                "latency_ms": latency_ms,
                "context_tokens": extra.get("context_tokens", 0),
                "retrieval": extra.get("retrieval", []),
            }
        )
    return rows


def _write_eval_outputs(output_dir: Path, rows: List[Dict[str, Any]]) -> None:
    save_results_jsonl(str(output_dir / "per_sample.jsonl"), rows)
    metrics = aggregate_metrics(rows)
    (output_dir / "aggregate_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    scalar_fields = [
        "sample_id",
        "dataset",
        "question",
        "reference_answer",
        "prediction",
        "exact_match",
        "nli_label",
        "nli_confidence",
        "hallucination_category",
        "latency_ms",
        "context_tokens",
    ]
    with (output_dir / "per_sample.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=scalar_fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in scalar_fields})


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Medical RAG-VLM research experiment runner")
    parser.add_argument("command", choices=["baseline-eval", "build-index", "rag-eval"])
    parser.add_argument("--config", default="configs/medical_rag_vlm.yaml", help="Path to experiment config")
    parser.add_argument(
        "--mode",
        default="hybrid",
        choices=["visual", "bm25", "hybrid"],
        help="Retrieval ablation mode for rag-eval",
    )
    return parser


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    args = build_arg_parser().parse_args()
    config = load_config(args.config)
    if args.command == "baseline-eval":
        output = run_baseline_eval(config)
    elif args.command == "build-index":
        output = run_build_index(config)
    else:
        output = run_rag_eval(config, mode=args.mode)
    print(f"Output: {output}")


if __name__ == "__main__":
    main()
