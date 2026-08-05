"""Evaluate fixed image classifiers under deterministic test-time degradation.

Training data and checkpoints are read-only. Results are saved after every
condition so an interrupted full-dataset experiment can be resumed safely.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import functional as TF

from run_gradcam_analysis import INPUT_SIZE, MEAN, MODEL_SPECS, STD, load_model


@dataclass(frozen=True)
class Condition:
    degradation: str
    severity: int
    parameter_name: str
    parameter_value: float


CONDITIONS = [Condition("clean", 0, "none", 0.0)]
CONDITIONS += [
    Condition("gaussian_noise", level, "sigma", value)
    for level, value in enumerate((0.03, 0.06, 0.12, 0.20), 1)
]
CONDITIONS += [
    Condition("gaussian_blur", level, "sigma", value)
    for level, value in enumerate((0.8, 1.5, 2.5, 4.0), 1)
]
CONDITIONS += [
    Condition("motion_blur", level, "kernel_size", value)
    for level, value in enumerate((3, 7, 11, 17), 1)
]
CONDITIONS += [
    Condition("jpeg_compression", level, "quality", value)
    for level, value in enumerate((80, 60, 40, 20), 1)
]


class DegradedImageDataset(Dataset):
    def __init__(
        self,
        root: Path,
        classes: list[str],
        condition: Condition,
        seed: int,
        max_images: int | None,
    ) -> None:
        self.root = root
        self.classes = classes
        self.condition = condition
        self.seed = seed
        local_classes = sorted(path.name for path in root.iterdir() if path.is_dir())
        if local_classes != classes:
            raise ValueError("Checkpoint classes do not exactly match test directories")
        items = [
            (path, class_index)
            for class_index, class_name in enumerate(classes)
            for path in sorted((root / class_name).glob("*.jpg"))
        ]
        if max_images is not None:
            # Evenly spaced selection avoids restricting a smoke test to early classes.
            if max_images <= 0:
                raise ValueError("max_images must be positive")
            if max_images < len(items):
                indices = torch.linspace(0, len(items) - 1, max_images).round().long()
                items = [items[index] for index in indices.tolist()]
        if not items:
            raise FileNotFoundError(f"No JPG test images found under {root}")
        self.items = items

    def __len__(self) -> int:
        return len(self.items)

    def _jpeg(self, image: Image.Image, quality: int) -> Image.Image:
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=quality, optimize=False)
        buffer.seek(0)
        with Image.open(buffer) as decoded:
            return decoded.convert("RGB")

    def _motion_blur(self, tensor: torch.Tensor, kernel_size: int) -> torch.Tensor:
        kernel = torch.ones(3, 1, 1, kernel_size, dtype=tensor.dtype) / kernel_size
        padded = F.pad(
            tensor.unsqueeze(0),
            (kernel_size // 2, kernel_size // 2, 0, 0),
            mode="reflect",
        )
        return F.conv2d(padded, kernel, groups=3).squeeze(0)

    def __getitem__(self, index: int):
        path, target = self.items[index]
        image = Image.open(path).convert("RGB")
        image = TF.center_crop(image, [INPUT_SIZE, INPUT_SIZE])
        condition = self.condition

        if condition.degradation == "jpeg_compression":
            image = self._jpeg(image, int(condition.parameter_value))

        tensor = TF.to_tensor(image)
        if condition.degradation == "gaussian_noise":
            generator = torch.Generator().manual_seed(self.seed + index)
            noise = torch.randn(tensor.shape, generator=generator, dtype=tensor.dtype)
            tensor = (tensor + noise * condition.parameter_value).clamp(0.0, 1.0)
        elif condition.degradation == "gaussian_blur":
            sigma = condition.parameter_value
            kernel_size = 2 * math.ceil(3 * sigma) + 1
            tensor = TF.gaussian_blur(tensor, [kernel_size, kernel_size], [sigma, sigma])
        elif condition.degradation == "motion_blur":
            tensor = self._motion_blur(tensor, int(condition.parameter_value))

        tensor = TF.normalize(tensor, mean=MEAN, std=STD)
        return tensor, target


def metrics_from_confusion(confusion: torch.Tensor) -> tuple[float, float]:
    confusion = confusion.to(torch.float64)
    total = confusion.sum()
    accuracy = float(confusion.diag().sum() / total) if total else 0.0
    true_positive = confusion.diag()
    false_positive = confusion.sum(dim=0) - true_positive
    false_negative = confusion.sum(dim=1) - true_positive
    denominator = 2 * true_positive + false_positive + false_negative
    f1 = torch.where(denominator > 0, 2 * true_positive / denominator, 0.0)
    return accuracy, float(f1.mean())


def evaluate(model, loader, num_classes: int, device: torch.device):
    confusion = torch.zeros((num_classes, num_classes), dtype=torch.int64)
    with torch.inference_mode():
        for images, targets in loader:
            predictions = model(images.to(device, non_blocking=True)).argmax(dim=1).cpu()
            encoded = targets * num_classes + predictions
            confusion += torch.bincount(
                encoded, minlength=num_classes * num_classes
            ).reshape(num_classes, num_classes)
    accuracy, macro_f1 = metrics_from_confusion(confusion)
    return accuracy, macro_f1, confusion


def write_rows(path: Path, rows: list[dict]) -> None:
    fieldnames = [
        "model",
        "checkpoint",
        "architecture",
        "degradation",
        "severity",
        "parameter_name",
        "parameter_value",
        "images",
        "top1_accuracy",
        "macro_f1",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def read_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoints", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-images", type=int)
    parser.add_argument("--models", nargs="*", choices=sorted(MODEL_SPECS))
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    checkpoint_dir = args.checkpoints.resolve()
    dataset_root = args.dataset.resolve()
    output_dir = args.output.resolve()
    if output_dir.exists() and not args.resume:
        raise FileExistsError(f"Refusing to overwrite existing output: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / "robustness_metrics.csv"
    rows = read_rows(results_path) if args.resume else []
    completed = {
        (row["checkpoint"], row["degradation"], int(row["severity"])) for row in rows
    }

    selected_names = args.models or list(MODEL_SPECS)
    missing = [name for name in selected_names if not (checkpoint_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing checkpoints: {missing}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    metadata = {
        "status": "running",
        "device": str(device),
        "seed": args.seed,
        "dataset": str(dataset_root),
        "checkpoints": str(checkpoint_dir),
        "batch_size": args.batch_size,
        "workers": args.workers,
        "max_images": args.max_images,
        "conditions": [condition.__dict__ for condition in CONDITIONS],
        "models": selected_names,
    }
    metadata_path = output_dir / "run_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    for checkpoint_name in selected_names:
        architecture, display_name = MODEL_SPECS[checkpoint_name]
        model, classes = load_model(checkpoint_dir / checkpoint_name, architecture, device)
        for condition in CONDITIONS:
            key = (checkpoint_name, condition.degradation, condition.severity)
            if key in completed:
                print(f"Skipping completed: {display_name} {condition}", flush=True)
                continue
            dataset = DegradedImageDataset(
                dataset_root, classes, condition, args.seed, args.max_images
            )
            loader = DataLoader(
                dataset,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=args.workers,
                pin_memory=device.type == "cuda",
                persistent_workers=args.workers > 0,
            )
            accuracy, macro_f1, confusion = evaluate(
                model, loader, len(classes), device
            )
            rows.append(
                {
                    "model": display_name,
                    "checkpoint": checkpoint_name,
                    "architecture": architecture,
                    "degradation": condition.degradation,
                    "severity": condition.severity,
                    "parameter_name": condition.parameter_name,
                    "parameter_value": condition.parameter_value,
                    "images": len(dataset),
                    "top1_accuracy": accuracy,
                    "macro_f1": macro_f1,
                }
            )
            write_rows(results_path, rows)
            confusion_dir = output_dir / "confusion_matrices"
            confusion_dir.mkdir(exist_ok=True)
            torch.save(
                confusion,
                confusion_dir
                / f"{Path(checkpoint_name).stem}_{condition.degradation}_s{condition.severity}.pt",
            )
            print(
                f"{display_name} | {condition.degradation} s{condition.severity} | "
                f"top1={accuracy:.4f} macro_f1={macro_f1:.4f}",
                flush=True,
            )
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    metadata["status"] = "completed"
    metadata["completed_conditions"] = len(rows)
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"Completed {len(rows)} model-condition evaluations", flush=True)


if __name__ == "__main__":
    main()
