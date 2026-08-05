"""Safely inspect PyTorch image-classification checkpoints.

The script is read-only and uses ``weights_only=True`` so that checkpoint
metadata can be checked without executing arbitrary pickled Python objects.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch


def describe(value: Any) -> dict[str, Any]:
    """Return compact, JSON-serialisable metadata for a checkpoint value."""
    if isinstance(value, dict):
        return {"type": "dict", "length": len(value)}
    if isinstance(value, (list, tuple)):
        preview = [str(item) for item in value[:3]]
        return {"type": type(value).__name__, "length": len(value), "preview": preview}
    if isinstance(value, torch.Tensor):
        return {
            "type": "tensor",
            "shape": list(value.shape),
            "dtype": str(value.dtype),
        }
    if isinstance(value, (str, int, float, bool)) or value is None:
        return {"type": type(value).__name__, "value": value}
    return {"type": type(value).__name__}


def inspect_checkpoint(path: Path) -> dict[str, Any]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict):
        raise TypeError(f"Expected a dictionary checkpoint, got {type(checkpoint).__name__}")

    state_dict = checkpoint.get("model_state_dict")
    if state_dict is None and all(isinstance(value, torch.Tensor) for value in checkpoint.values()):
        state_dict = checkpoint

    parameter_shapes: dict[str, list[int]] = {}
    if isinstance(state_dict, dict):
        parameter_shapes = {
            name: list(tensor.shape)
            for name, tensor in state_dict.items()
            if isinstance(tensor, torch.Tensor)
        }

    classes = checkpoint.get("classes")
    return {
        "file": path.name,
        "size_bytes": path.stat().st_size,
        "keys": sorted(checkpoint.keys()),
        "metadata": {
            key: describe(value)
            for key, value in checkpoint.items()
            if key != "model_state_dict"
        },
        "declared_num_classes": checkpoint.get("num_classes"),
        "class_count": len(classes) if isinstance(classes, (list, tuple)) else None,
        "parameter_count": len(parameter_shapes),
        "first_parameter_shapes": dict(list(parameter_shapes.items())[:3]),
        "last_parameter_shapes": dict(list(parameter_shapes.items())[-3:]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint_dir", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    checkpoint_dir = args.checkpoint_dir.resolve()
    paths = sorted(checkpoint_dir.glob("*.pth"))
    if not paths:
        raise FileNotFoundError(f"No .pth files found in {checkpoint_dir}")

    report = [inspect_checkpoint(path) for path in paths]
    rendered = json.dumps(report, indent=2)
    print(rendered)

    if args.output:
        output = args.output.resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        if output.exists():
            raise FileExistsError(f"Refusing to overwrite existing report: {output}")
        output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
