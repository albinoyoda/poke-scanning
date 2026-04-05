"""Export MobileNetV3-Small feature extractor to ONNX format.

Usage:
    uv run python scripts/export_model.py [--output data/mobilenet_v3_small.onnx]

Requires dev dependencies (torch, torchvision).  Run once to produce
the ONNX file; torch is not needed at runtime.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torchvision.models as models


def main() -> None:
    parser = argparse.ArgumentParser(description="Export MobileNetV3-Small to ONNX")
    parser.add_argument(
        "--output",
        type=str,
        default="data/mobilenet_v3_small.onnx",
        help="Output ONNX file path",
    )
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print("Loading MobileNetV3-Small (ImageNet-1k weights)...")
    model = models.mobilenet_v3_small(weights="IMAGENET1K_V1")

    # Remove classifier head — keep only the feature extractor + avgpool.
    # The original classifier is: Sequential(Linear(576,1024), Hardswish,
    # Dropout, Linear(1024,1000)).  We replace it with Identity so the
    # output is the 576-dim feature vector after global average pooling.
    model.classifier = torch.nn.Identity()
    model.eval()

    dummy_input = torch.randn(1, 3, 224, 224)

    print(f"Exporting to {output_path}...")
    torch.onnx.export(
        model,
        dummy_input,
        str(output_path),
        input_names=["input"],
        output_names=["embedding"],
        dynamic_axes={"input": {0: "batch"}, "embedding": {0: "batch"}},
        opset_version=17,
    )

    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"Done. Model size: {size_mb:.1f} MB")

    # Verify the exported model produces correct output shape.
    import onnxruntime as ort

    session = ort.InferenceSession(str(output_path))
    result = session.run(None, {"input": dummy_input.numpy()})
    print(f"Output shape: {result[0].shape}")  # Should be (1, 576)


if __name__ == "__main__":
    main()
