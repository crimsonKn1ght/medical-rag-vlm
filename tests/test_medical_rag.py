import json
import tempfile
import unittest
from pathlib import Path

from medical_rag.backends import StubMedicalVLMBackend
from medical_rag.data import load_mimic_cxr_corpus, load_vqa_dataset, parse_report_sections
from medical_rag.metrics import exact_match
from medical_rag.prompts import build_context
from medical_rag.retrieval import BM25Index, LexicalOverlapReranker, build_visual_index, hybrid_search
from medical_rag.runner import run_baseline_eval, run_build_index, run_rag_eval


def write_image(path: Path, color):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"synthetic image placeholder: {color}", encoding="utf-8")


class MedicalRAGTests(unittest.TestCase):
    def test_report_sections_and_vqa_loading(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_image(root / "images" / "a.png", "white")
            vqa_path = root / "vqa.json"
            vqa_path.write_text(
                json.dumps([{"id": "q1", "image": "a.png", "question": "Finding?", "answer": "pneumonia"}]),
                encoding="utf-8",
            )
            samples = load_vqa_dataset(str(vqa_path), image_root=str(root / "images"), dataset="vqa_rad")
            self.assertEqual(samples[0].sample_id, "q1")
            self.assertTrue(samples[0].image_path.endswith("a.png"))

        sections = parse_report_sections("FINDINGS:\nClear lungs.\nIMPRESSION:\nNormal chest.")
        self.assertEqual(sections["findings"], "Clear lungs.")
        self.assertEqual(sections["impression"], "Normal chest.")

    def test_retrieval_and_context(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_image(root / "img1.png", "red")
            write_image(root / "img2.png", "blue")
            manifest = root / "manifest.json"
            manifest.write_text(
                json.dumps(
                    [
                        {
                            "case_id": "c1",
                            "image_path": "img1.png",
                            "report_text": "FINDINGS:\nRight lower lobe pneumonia.\nIMPRESSION:\nPneumonia.",
                        },
                        {
                            "case_id": "c2",
                            "image_path": "img2.png",
                            "report_text": "FINDINGS:\nClear lungs.\nIMPRESSION:\nNormal chest.",
                        },
                    ]
                ),
                encoding="utf-8",
            )
            cases = load_mimic_cxr_corpus(str(manifest), image_root=str(root))
            bm25 = BM25Index(cases)
            bm25_hits = bm25.search("pneumonia", top_k=1)
            self.assertEqual(bm25_hits[0].case_id, "c1")

            backend = StubMedicalVLMBackend()
            visual = build_visual_index(cases, backend)
            query_vector = backend.encode_image(str(root / "img1.png"))
            hits = hybrid_search("pneumonia", query_vector, bm25, visual, 2, 2, 0.5, 0.5, LexicalOverlapReranker())
            self.assertEqual(hits[0].case_id, "c1")
            context, used = build_context(hits, token_budget=30)
            self.assertIn("Retrieved case", context)
            self.assertLessEqual(used, 30)

    def test_metrics(self):
        self.assertEqual(exact_match("The Pneumonia.", "pneumonia"), 1.0)

    def test_runner_smoke(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_image(root / "vqa_images" / "q.png", "red")
            write_image(root / "cxr_images" / "case.png", "red")
            (root / "reports").mkdir()
            (root / "reports" / "case.txt").write_text(
                "FINDINGS:\nRight lower lobe pneumonia.\nIMPRESSION:\nPneumonia.",
                encoding="utf-8",
            )
            vqa_path = root / "vqa.json"
            vqa_path.write_text(
                json.dumps([{"id": "q1", "image": "q.png", "question": "What disease is present?", "answer": "pneumonia"}]),
                encoding="utf-8",
            )
            manifest = root / "manifest.csv"
            manifest.write_text("case_id,image_path,report_path\nc1,case.png,case.txt\n", encoding="utf-8")
            config = {
                "model": {"type": "stub", "embedding_dim": 32},
                "datasets": {
                    "vqa": [{"name": "fixture_vqa", "path": str(vqa_path), "image_root": str(root / "vqa_images")}],
                    "mimic_cxr": {
                        "manifest_path": str(manifest),
                        "image_root": str(root / "cxr_images"),
                        "report_root": str(root / "reports"),
                    },
                },
                "retrieval": {
                    "index_dir": str(root / "indexes"),
                    "top_k": 1,
                    "candidate_pool_size": 1,
                    "visual_weight": 0.5,
                    "text_weight": 0.5,
                    "rerank": True,
                },
                "prompting": {"context_token_budget": 50},
                "outputs": {"root_dir": str(root / "outputs")},
            }
            baseline_dir = run_baseline_eval(config)
            index_dir = run_build_index(config)
            rag_dir = run_rag_eval(config, mode="hybrid")

            partial_rows = (baseline_dir / "per_sample.partial.jsonl").read_text(encoding="utf-8").strip().splitlines()
            self.assertEqual(len(partial_rows), 1)
            self.assertTrue((baseline_dir / "per_sample.jsonl").exists())
            self.assertTrue((index_dir / "manifest.json").exists())
            self.assertTrue((rag_dir / "aggregate_metrics.json").exists())


if __name__ == "__main__":
    unittest.main()
