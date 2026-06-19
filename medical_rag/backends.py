import hashlib
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


class MedicalVLMBackend(ABC):
    """Frozen medical VLM interface used by the research runner."""

    @abstractmethod
    def generate(self, image_path: str, question: str, context: Optional[str] = None) -> str:
        raise NotImplementedError

    @abstractmethod
    def encode_image(self, image_path: str, mode: str = "global") -> np.ndarray:
        raise NotImplementedError


class StubMedicalVLMBackend(MedicalVLMBackend):
    """Deterministic local backend for tests, fixtures, and dry runs."""

    def __init__(self, embedding_dim: int = 32):
        self.embedding_dim = embedding_dim

    def generate(self, image_path: str, question: str, context: Optional[str] = None) -> str:
        text = f"{question}\n{context or ''}".lower()
        if "pneumonia" in text:
            return "pneumonia"
        if "cardiomegaly" in text or "enlarged heart" in text:
            return "cardiomegaly"
        if "normal" in text or "clear lungs" in text:
            return "normal"
        return "no acute abnormality"

    def encode_image(self, image_path: str, mode: str = "global") -> np.ndarray:
        path = Path(image_path)
        digest = hashlib.sha256(str(path).encode("utf-8")).digest()
        vector = np.frombuffer(digest, dtype=np.uint8).astype(np.float32)
        while vector.size < self.embedding_dim:
            vector = np.concatenate([vector, vector])
        vector = vector[: self.embedding_dim]
        norm = np.linalg.norm(vector) or 1.0
        if mode == "patch":
            return np.stack([vector / norm, np.roll(vector, 1) / norm], axis=0)
        return vector / norm


class HuggingFaceMedicalVLMBackend(MedicalVLMBackend):
    """Best-effort wrapper for Hugging Face-compatible frozen medical VLMs."""

    def __init__(self, config: Dict[str, Any]):
        try:
            import torch
            from transformers import AutoModelForImageTextToText, AutoProcessor
        except ImportError as exc:
            raise ImportError("transformers and torch are required for HuggingFaceMedicalVLMBackend") from exc

        self.torch = torch
        self.model_id = config["model_id"]
        self.device = config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        self.max_new_tokens = int(config.get("max_new_tokens", 64))
        self.trust_remote_code = bool(config.get("trust_remote_code", True))
        dtype_name = config.get("torch_dtype", "bfloat16")
        dtype = getattr(torch, dtype_name, torch.bfloat16)

        self.processor = AutoProcessor.from_pretrained(
            self.model_id,
            trust_remote_code=self.trust_remote_code,
        )
        self.model = AutoModelForImageTextToText.from_pretrained(
            self.model_id,
            torch_dtype=dtype,
            trust_remote_code=self.trust_remote_code,
        ).to(self.device)
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

    def _prompt(self, question: str, context: Optional[str]) -> str:
        if context:
            return (
                "Use the medical image and the retrieved evidence below. "
                "Answer only what is supported.\n\n"
                f"Retrieved evidence:\n{context}\n\nQuestion: {question}"
            )
        return f"Question: {question}"

    def generate(self, image_path: str, question: str, context: Optional[str] = None) -> str:
        from PIL import Image

        image = Image.open(image_path).convert("RGB")
        prompt = self._prompt(question, context)
        inputs = self.processor(images=image, text=prompt, return_tensors="pt").to(self.device)
        with self.torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                temperature=None,
            )
        return self.processor.batch_decode(output_ids, skip_special_tokens=True)[0].strip()

    def encode_image(self, image_path: str, mode: str = "global") -> np.ndarray:
        from PIL import Image

        image = Image.open(image_path).convert("RGB")
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        with self.torch.no_grad():
            if hasattr(self.model, "get_image_features"):
                features = self.model.get_image_features(**inputs)
            else:
                logger.warning("Model does not expose get_image_features; using deterministic path hash embedding")
                return StubMedicalVLMBackend().encode_image(image_path, mode=mode)
        array = features.detach().float().cpu().numpy()
        if mode == "global" and array.ndim > 1:
            array = array.reshape(-1, array.shape[-1]).mean(axis=0)
        norm = np.linalg.norm(array, axis=-1, keepdims=True)
        return array / np.maximum(norm, 1e-12)


class CustomConnectorBaselineBackend(MedicalVLMBackend):
    """Adapter for the repo's original CLIP+Qwen connector baseline."""

    def __init__(self, config: Dict[str, Any]):
        try:
            import torch

            from data.image_processing import load_and_process_image
            from inference import load_vlm, run_inference
        except ImportError as exc:
            raise ImportError("The custom connector baseline requires the original VLM dependencies") from exc

        self.torch = torch
        self.load_and_process_image = load_and_process_image
        self.run_inference = run_inference
        self.device = config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        self.model = load_vlm(
            config_path=config["config_path"],
            connector_checkpoint=config["checkpoint"],
            device=self.device,
        )

    def generate(self, image_path: str, question: str, context: Optional[str] = None) -> str:
        prompt = question if not context else f"Retrieved evidence:\n{context}\n\nQuestion: {question}"
        return self.run_inference(
            model=self.model,
            image_path=image_path,
            prompt=prompt,
            max_new_tokens=64,
            temperature=0.0,
            device=self.device,
        )

    def encode_image(self, image_path: str, mode: str = "global") -> np.ndarray:
        pixel_values = self.load_and_process_image(image_path, self.model.image_processor)
        pixel_values = pixel_values.unsqueeze(0).to(self.device)
        with self.torch.no_grad():
            image_embeds = self.model.encode_images(pixel_values).detach().float().cpu().numpy()[0]
        if mode == "patch":
            return image_embeds
        vector = image_embeds.mean(axis=0)
        return vector / max(float(np.linalg.norm(vector)), 1e-12)


def build_backend(config: Dict[str, Any]) -> MedicalVLMBackend:
    backend_type = config.get("type", "huggingface")
    if backend_type == "stub":
        return StubMedicalVLMBackend(embedding_dim=int(config.get("embedding_dim", 32)))
    if backend_type == "huggingface":
        return HuggingFaceMedicalVLMBackend(config)
    if backend_type == "custom_connector_baseline":
        return CustomConnectorBaselineBackend(config)
    raise ValueError(f"Unsupported medical VLM backend type: {backend_type}")
