"""CNN embedding extraction via ONNX Runtime.

Loads a MobileNetV3-Small ONNX model (exported by scripts/export_model.py)
and produces L2-normalised 576-dim float32 embeddings for card images.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
from numpy.typing import NDArray
from PIL import Image

EMBEDDING_DIM = 576

DEFAULT_MODEL_PATH = Path("data") / "mobilenet_v3_small.onnx"

# ImageNet normalisation constants.
_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

_INPUT_SIZE = 224


def _preprocess(image_rgb: NDArray[np.uint8]) -> NDArray[np.float32]:
    """Convert an RGB uint8 image to a normalised NCHW float32 tensor.

    Steps:
    1. Resize to 224x224 (bilinear).
    2. Scale to [0, 1].
    3. Normalise with ImageNet mean/std.
    4. Transpose to NCHW layout: (1, 3, 224, 224).
    """
    resized = cv2.resize(image_rgb, (_INPUT_SIZE, _INPUT_SIZE))
    tensor = resized.astype(np.float32) / 255.0
    tensor = (tensor - _IMAGENET_MEAN) / _IMAGENET_STD
    # HWC → CHW → NCHW
    tensor = np.transpose(tensor, (2, 0, 1))[np.newaxis]
    return np.ascontiguousarray(tensor)


def _l2_normalise(vectors: NDArray[np.float32]) -> NDArray[np.float32]:
    """L2-normalise each row of *vectors*."""
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-10)
    return vectors / norms


class CardEmbedder:
    """Produces CNN embeddings for card images via ONNX Runtime."""

    def __init__(self, model_path: Path | str | None = None) -> None:
        path = Path(model_path) if model_path else DEFAULT_MODEL_PATH
        if not path.exists():
            raise FileNotFoundError(
                f"ONNX model not found: {path}\n"
                "Run: uv run python scripts/export_model.py"
            )
        self._session = ort.InferenceSession(
            str(path),
            providers=["CPUExecutionProvider"],
        )
        self._input_name = self._session.get_inputs()[0].name

    def embed(self, image: NDArray[np.uint8]) -> NDArray[np.float32]:
        """Compute embedding for a BGR card image.

        Returns an L2-normalised float32 vector of shape ``(576,)``.
        """
        rgb = image[..., ::-1]  # BGR → RGB
        tensor = _preprocess(rgb)
        (output,) = self._session.run(None, {self._input_name: tensor})
        arr: NDArray[np.float32] = np.asarray(output)
        return _l2_normalise(arr.reshape(1, -1))[0]

    def embed_pil(self, pil_image: Image.Image) -> NDArray[np.float32]:
        """Compute embedding for a PIL RGB Image."""
        rgb = np.asarray(pil_image, dtype=np.uint8)
        if rgb.ndim == 2:
            rgb = np.stack([rgb] * 3, axis=-1)
        tensor = _preprocess(rgb)
        (output,) = self._session.run(None, {self._input_name: tensor})
        arr = np.asarray(output)
        return _l2_normalise(arr.reshape(1, -1))[0]

    def embed_batch(self, images: list[NDArray[np.uint8]]) -> NDArray[np.float32]:
        """Compute embeddings for a batch of BGR card images.

        Returns L2-normalised float32 vectors of shape ``(N, 576)``.
        """
        if not images:
            return np.empty((0, EMBEDDING_DIM), dtype=np.float32)

        tensors = []
        for img in images:
            rgb = img[..., ::-1]
            tensors.append(_preprocess(rgb)[0])  # drop batch dim
        batch = np.stack(tensors)  # (N, 3, 224, 224)
        (output,) = self._session.run(None, {self._input_name: batch})
        arr = np.asarray(output)
        return _l2_normalise(arr.reshape(len(images), -1))
