import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


@dataclass
class VQASample:
    sample_id: str
    image_path: str
    question: str
    answer: str
    dataset: str = ""
    meta: Dict[str, Any] = None

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["meta"] = self.meta or {}
        return payload


@dataclass
class CorpusCase:
    case_id: str
    image_path: str
    report_path: str
    report_text: str
    findings: str = ""
    impression: str = ""
    meta: Dict[str, Any] = None

    @property
    def retrieval_text(self) -> str:
        priority = "\n".join(x for x in [self.findings, self.impression] if x)
        return priority or self.report_text

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["meta"] = self.meta or {}
        return payload


def read_json_or_jsonl(path: str) -> List[Dict[str, Any]]:
    source = Path(path)
    if source.suffix.lower() == ".jsonl":
        with source.open("r", encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]
    with source.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        for key in ("samples", "data", "annotations"):
            if isinstance(data.get(key), list):
                return data[key]
        return [data]
    return data


def load_vqa_dataset(path: str, image_root: Optional[str] = None, dataset: str = "") -> List[VQASample]:
    rows = read_json_or_jsonl(path)
    samples: List[VQASample] = []
    root = Path(image_root) if image_root else None

    for idx, row in enumerate(rows):
        sample_id = str(
            row.get("id")
            or row.get("qid")
            or row.get("question_id")
            or row.get("sample_id")
            or f"{dataset or 'vqa'}_{idx}"
        )
        image_value = row.get("image") or row.get("image_path") or row.get("img_name")
        if image_value is None:
            raise ValueError(f"VQA sample {sample_id} is missing an image path")
        image_path = Path(str(image_value))
        if root and not image_path.is_absolute():
            image_path = root / image_path

        question = row.get("question") or row.get("query")
        answer = row.get("answer") or row.get("answer_text") or row.get("label")
        if question is None or answer is None:
            raise ValueError(f"VQA sample {sample_id} is missing question or answer")

        samples.append(
            VQASample(
                sample_id=sample_id,
                image_path=str(image_path),
                question=str(question),
                answer=str(answer),
                dataset=dataset,
                meta={k: v for k, v in row.items() if k not in {"image", "image_path", "question", "answer"}},
            )
        )
    return samples


def parse_report_sections(report_text: str) -> Dict[str, str]:
    sections = {"findings": "", "impression": ""}
    current = None
    buffers = {"findings": [], "impression": []}

    for raw_line in report_text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        label = line.rstrip(":").lower()
        if label in buffers:
            current = label
            remainder = line[len(label) :].lstrip(" :")
            if remainder:
                buffers[current].append(remainder)
            continue
        if current:
            buffers[current].append(line)

    for key, values in buffers.items():
        sections[key] = " ".join(values).strip()
    return sections


def load_mimic_cxr_corpus(manifest_path: str, image_root: Optional[str] = None, report_root: Optional[str] = None) -> List[CorpusCase]:
    manifest = Path(manifest_path)
    if manifest.suffix.lower() == ".csv":
        with manifest.open("r", encoding="utf-8", newline="") as f:
            rows: Iterable[Dict[str, Any]] = list(csv.DictReader(f))
    else:
        rows = read_json_or_jsonl(str(manifest))

    image_base = Path(image_root) if image_root else None
    report_base = Path(report_root) if report_root else None
    cases: List[CorpusCase] = []

    for idx, row in enumerate(rows):
        case_id = str(row.get("case_id") or row.get("study_id") or row.get("id") or f"case_{idx}")
        image_value = row.get("image_path") or row.get("image") or row.get("dicom_path")
        report_value = row.get("report_path") or row.get("report") or row.get("text_path")
        report_text = str(row.get("report_text") or row.get("text") or "")

        if not image_value:
            raise ValueError(f"Corpus case {case_id} is missing image_path")
        image_path = Path(str(image_value))
        if image_base and not image_path.is_absolute():
            image_path = image_base / image_path

        report_path = Path(str(report_value or ""))
        if report_value and report_base and not report_path.is_absolute():
            report_path = report_base / report_path
        if not report_text and report_value:
            report_text = report_path.read_text(encoding="utf-8")
        if not report_text:
            raise ValueError(f"Corpus case {case_id} is missing report text")

        sections = parse_report_sections(report_text)
        cases.append(
            CorpusCase(
                case_id=case_id,
                image_path=str(image_path),
                report_path=str(report_path),
                report_text=report_text,
                findings=sections["findings"],
                impression=sections["impression"],
                meta={k: v for k, v in row.items() if k not in {"report_text", "text"}},
            )
        )
    return cases
