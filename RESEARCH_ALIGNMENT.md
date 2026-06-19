# Medical RAG-VLM Research Alignment

This repo now has two paths:

- The original `train.py` / `inference.py` connector-training code remains available as a custom LLaVA-style baseline.
- The research statement path is implemented through `medical_rag_experiments.py`, which evaluates frozen medical VLMs with and without retrieval grounding.

## Dataset Layout

Medical datasets are expected as local files because MIMIC-CXR is gated and benchmark distributions vary.

VQA files may be JSON or JSONL with fields such as:

```json
{"id": "sample-1", "image": "case.png", "question": "What abnormality is present?", "answer": "pneumonia"}
```

MIMIC-CXR corpus manifests may be CSV, JSON, or JSONL with:

```csv
case_id,image_path,report_path
study-1,p10/study-1.png,p10/study-1.txt
```

Reports should contain plain text. When `FINDINGS:` and `IMPRESSION:` sections exist, those sections are prioritized in retrieved context.

## Commands

Create retrieval indexes:

```bash
python medical_rag_experiments.py build-index --config configs/medical_rag_vlm.yaml
```

Run baseline frozen-medical-VLM evaluation:

```bash
python medical_rag_experiments.py baseline-eval --config configs/medical_rag_vlm.yaml
```

Run retrieval ablations:

```bash
python medical_rag_experiments.py rag-eval --mode visual --config configs/medical_rag_vlm.yaml
python medical_rag_experiments.py rag-eval --mode bm25 --config configs/medical_rag_vlm.yaml
python medical_rag_experiments.py rag-eval --mode hybrid --config configs/medical_rag_vlm.yaml
```

Each run writes `per_sample.jsonl`, `per_sample.csv`, and `aggregate_metrics.json` under `outputs/`.

## Research Phase Mapping

- Phase 1, baseline hallucination characterization: `baseline-eval` on VQA-RAD and PathVQA with exact-match and NLI-style factual consistency outputs.
- Phase 2, retrieval construction and ablation: `build-index`, then `rag-eval` with `visual`, `bm25`, and `hybrid` modes.
- Phase 3, architecture refinement: adjust `context_token_budget`, `visual_weight`, `text_weight`, and reranking in `configs/medical_rag_vlm.yaml`; add query decomposition behind the same prompt/context interfaces.

## Notes

- `faiss` is used automatically when installed; otherwise dense retrieval falls back to NumPy cosine search for local tests.
- The included NLI scorer has a deterministic fallback so tests can run offline. Replace it with a configured clinical or general NLI model for publication runs.
- Do not commit clinical data or generated outputs.
