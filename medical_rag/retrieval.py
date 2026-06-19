import json
import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np

from .data import CorpusCase


TOKEN_RE = re.compile(r"[a-z0-9]+")


@dataclass
class RetrievalResult:
    case_id: str
    score: float
    rank: int
    source: str
    image_path: str = ""
    report_text: str = ""
    findings: str = ""
    impression: str = ""

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


def tokenize(text: str) -> List[str]:
    return TOKEN_RE.findall(text.lower())


class BM25Index:
    def __init__(self, cases: Sequence[CorpusCase], k1: float = 1.5, b: float = 0.75):
        self.cases = list(cases)
        self.k1 = k1
        self.b = b
        self.docs = [tokenize(case.retrieval_text) for case in self.cases]
        self.doc_lens = [len(doc) for doc in self.docs]
        self.avgdl = sum(self.doc_lens) / max(len(self.doc_lens), 1)
        self.df: Dict[str, int] = {}
        for doc in self.docs:
            for term in set(doc):
                self.df[term] = self.df.get(term, 0) + 1

    def search(self, query: str, top_k: int) -> List[RetrievalResult]:
        query_terms = tokenize(query)
        scores = []
        total_docs = max(len(self.docs), 1)
        for idx, doc in enumerate(self.docs):
            tf: Dict[str, int] = {}
            for term in doc:
                tf[term] = tf.get(term, 0) + 1
            score = 0.0
            for term in query_terms:
                if term not in tf:
                    continue
                idf = math.log(1 + (total_docs - self.df.get(term, 0) + 0.5) / (self.df.get(term, 0) + 0.5))
                denom = tf[term] + self.k1 * (1 - self.b + self.b * self.doc_lens[idx] / max(self.avgdl, 1e-12))
                score += idf * (tf[term] * (self.k1 + 1)) / max(denom, 1e-12)
            if score > 0:
                scores.append((idx, score))
        scores.sort(key=lambda item: item[1], reverse=True)
        return [_result_from_case(self.cases[idx], score, rank + 1, "bm25") for rank, (idx, score) in enumerate(scores[:top_k])]

    def save(self, path: str) -> None:
        payload = {"cases": [case.to_dict() for case in self.cases], "k1": self.k1, "b": self.b}
        Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str) -> "BM25Index":
        from .data import CorpusCase

        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        cases = [CorpusCase(**item) for item in payload["cases"]]
        return cls(cases, k1=payload.get("k1", 1.5), b=payload.get("b", 0.75))


class DenseImageIndex:
    def __init__(self, cases: Sequence[CorpusCase], vectors: np.ndarray):
        self.cases = list(cases)
        self.vectors = _normalize(np.asarray(vectors, dtype=np.float32))
        self.faiss_index = None
        try:
            import faiss

            index = faiss.IndexFlatIP(self.vectors.shape[1])
            index.add(self.vectors)
            self.faiss_index = index
        except Exception:
            self.faiss_index = None

    def search(self, query_vector: np.ndarray, top_k: int) -> List[RetrievalResult]:
        query = _normalize(np.asarray(query_vector, dtype=np.float32).reshape(1, -1))
        if self.faiss_index is not None:
            scores, indices = self.faiss_index.search(query, top_k)
            pairs = [(int(i), float(s)) for i, s in zip(indices[0], scores[0]) if i >= 0]
        else:
            sims = self.vectors @ query[0]
            order = np.argsort(-sims)[:top_k]
            pairs = [(int(i), float(sims[i])) for i in order]
        return [_result_from_case(self.cases[idx], score, rank + 1, "visual") for rank, (idx, score) in enumerate(pairs)]

    def save(self, output_dir: str) -> None:
        target = Path(output_dir)
        target.mkdir(parents=True, exist_ok=True)
        np.save(target / "visual_vectors.npy", self.vectors)
        (target / "visual_cases.json").write_text(
            json.dumps([case.to_dict() for case in self.cases], indent=2),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, output_dir: str) -> "DenseImageIndex":
        from .data import CorpusCase

        target = Path(output_dir)
        vectors = np.load(target / "visual_vectors.npy")
        cases = [CorpusCase(**item) for item in json.loads((target / "visual_cases.json").read_text(encoding="utf-8"))]
        return cls(cases, vectors)


class LexicalOverlapReranker:
    def rerank(self, query: str, results: Sequence[RetrievalResult], top_k: int) -> List[RetrievalResult]:
        q_terms = set(tokenize(query))
        rescored = []
        for result in results:
            text_terms = set(tokenize(" ".join([result.findings, result.impression, result.report_text])))
            overlap = len(q_terms & text_terms) / max(len(q_terms), 1)
            rescored.append((result, result.score + overlap))
        rescored.sort(key=lambda item: item[1], reverse=True)
        output = []
        for rank, (result, score) in enumerate(rescored[:top_k], start=1):
            output.append(
                RetrievalResult(
                    case_id=result.case_id,
                    score=float(score),
                    rank=rank,
                    source=result.source,
                    image_path=result.image_path,
                    report_text=result.report_text,
                    findings=result.findings,
                    impression=result.impression,
                )
            )
        return output


def build_visual_index(cases: Sequence[CorpusCase], backend) -> DenseImageIndex:
    vectors = []
    for case in cases:
        vector = backend.encode_image(case.image_path, mode="global")
        vectors.append(np.asarray(vector).reshape(-1))
    return DenseImageIndex(cases, np.stack(vectors, axis=0))


def hybrid_search(
    query: str,
    query_vector: np.ndarray,
    bm25_index: Optional[BM25Index],
    visual_index: Optional[DenseImageIndex],
    top_k: int,
    candidate_pool_size: int,
    visual_weight: float,
    text_weight: float,
    reranker: Optional[LexicalOverlapReranker] = None,
) -> List[RetrievalResult]:
    merged: Dict[str, RetrievalResult] = {}

    if visual_index is not None:
        for result in visual_index.search(query_vector, candidate_pool_size):
            result.score *= visual_weight
            merged[result.case_id] = result

    if bm25_index is not None:
        for result in bm25_index.search(query, candidate_pool_size):
            weighted = result.score * text_weight
            existing = merged.get(result.case_id)
            if existing:
                existing.score += weighted
                existing.source = "hybrid"
            else:
                result.score = weighted
                merged[result.case_id] = result

    results = sorted(merged.values(), key=lambda item: item.score, reverse=True)
    for rank, result in enumerate(results, start=1):
        result.rank = rank
    if reranker:
        return reranker.rerank(query, results, top_k)
    return results[:top_k]


def save_results_jsonl(path: str, rows: Iterable[Dict[str, object]]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _normalize(array: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(array, axis=-1, keepdims=True)
    return array / np.maximum(norm, 1e-12)


def _result_from_case(case: CorpusCase, score: float, rank: int, source: str) -> RetrievalResult:
    return RetrievalResult(
        case_id=case.case_id,
        score=float(score),
        rank=rank,
        source=source,
        image_path=case.image_path,
        report_text=case.report_text,
        findings=case.findings,
        impression=case.impression,
    )
