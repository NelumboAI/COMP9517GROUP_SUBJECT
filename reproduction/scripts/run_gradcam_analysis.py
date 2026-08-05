"""Generate deterministic, report-ready Grad-CAM comparisons.

The script selects confident correct and incorrect examples with the best
pretrained model, then compares every supplied 500-class checkpoint on the
same images. It never modifies checkpoints or dataset files and refuses to
overwrite an existing output directory.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms
from torchvision.transforms import functional as TF


INPUT_SIZE = 320
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
REFERENCE_FILE = "inat_resnet50_500_ptclasses_imagenet.pth"
MODEL_SPECS = {
    "inat_alexnet_500_npclasses_imagenet.pth": ("alexnet", "AlexNet random"),
    "inat_alexnet_500_ptclasses_imagenet.pth": ("alexnet", "AlexNet pretrained"),
    "inat_resnet18_500_npclasses_imagenet.pth": ("resnet18", "ResNet18 random"),
    "inat_resnet18_500_ptclasses_imagenet.pth": ("resnet18", "ResNet18 pretrained"),
    "inat_resnet50_500_npclasses_imagenet.pth": ("resnet50", "ResNet50 random"),
    "inat_resnet50_500_ptclasses_imagenet.pth": ("resnet50", "ResNet50 pretrained"),
}


TRANSFORM = transforms.Compose(
    [
        transforms.CenterCrop(INPUT_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD),
    ]
)


def create_model(architecture: str, num_classes: int) -> nn.Module:
    if architecture == "alexnet":
        model = models.alexnet(weights=None)
        model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)
    elif architecture == "resnet18":
        model = models.resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif architecture == "resnet50":
        model = models.resnet50(weights=None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    else:
        raise ValueError(f"Unsupported architecture: {architecture}")
    return model


def load_model(path: Path, architecture: str, device: torch.device):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    classes = checkpoint["classes"]
    if checkpoint["num_classes"] != len(classes):
        raise ValueError(f"Inconsistent class metadata in {path.name}")
    model = create_model(architecture, len(classes))
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    return model.to(device).eval(), list(classes)


def dataset_items(dataset_root: Path, classes: list[str]) -> list[tuple[Path, int]]:
    local_classes = sorted(path.name for path in dataset_root.iterdir() if path.is_dir())
    if local_classes != classes:
        raise ValueError("Checkpoint classes do not exactly match dataset directories")
    items = []
    for class_index, class_name in enumerate(classes):
        for path in sorted((dataset_root / class_name).glob("*.jpg")):
            items.append((path, class_index))
    if not items:
        raise FileNotFoundError(f"No JPG images found under {dataset_root}")
    return items


def predict(model: nn.Module, image_path: Path, device: torch.device):
    image = Image.open(image_path).convert("RGB")
    tensor = TRANSFORM(image).unsqueeze(0).to(device)
    with torch.inference_mode():
        probabilities = torch.softmax(model(tensor), dim=1)[0]
    confidence, prediction = probabilities.max(dim=0)
    return int(prediction.item()), float(confidence.item())


def target_layer(model: nn.Module, architecture: str) -> nn.Module:
    return model.features[10] if architecture == "alexnet" else model.layer4[-1]


def gradcam(model: nn.Module, architecture: str, tensor: torch.Tensor):
    activations = {}
    gradients = {}

    def hook(_module, _inputs, output):
        activations["value"] = output
        output.register_hook(lambda gradient: gradients.setdefault("value", gradient))

    handle = target_layer(model, architecture).register_forward_hook(hook)
    try:
        model.zero_grad(set_to_none=True)
        with torch.enable_grad():
            logits = model(tensor)
            prediction = int(logits.argmax(dim=1).item())
            logits[0, prediction].backward()
        feature_maps = activations["value"][0]
        feature_gradients = gradients["value"][0]
        weights = feature_gradients.mean(dim=(1, 2), keepdim=True)
        heatmap = torch.relu((weights * feature_maps).sum(dim=0))
        heatmap -= heatmap.min()
        if heatmap.max().item() > 0:
            heatmap /= heatmap.max()
        probability = torch.softmax(logits.detach(), dim=1)[0, prediction]
        return heatmap.detach().cpu().numpy(), prediction, float(probability.item())
    finally:
        handle.remove()


def short_label(class_name: str) -> str:
    parts = class_name.split("_")
    return "_".join(parts[-2:]) if len(parts) >= 2 else class_name


def select_examples(model, items, device, scan_limit, per_group, seed):
    rng = random.Random(seed)
    candidates = items.copy()
    rng.shuffle(candidates)
    candidates = candidates[: min(scan_limit, len(candidates))]
    correct, incorrect = [], []
    for path, truth in candidates:
        prediction, confidence = predict(model, path, device)
        record = (confidence, path, truth, prediction)
        (correct if prediction == truth else incorrect).append(record)
    correct.sort(key=lambda row: (-row[0], str(row[1])))
    incorrect.sort(key=lambda row: (-row[0], str(row[1])))
    if len(correct) < per_group or len(incorrect) < per_group:
        raise RuntimeError(
            f"Scanned {len(candidates)} images but found only "
            f"{len(correct)} correct and {len(incorrect)} incorrect examples"
        )
    return correct[:per_group] + incorrect[:per_group]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoints", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scan-limit", type=int, default=300)
    parser.add_argument("--examples-per-group", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    checkpoint_dir = args.checkpoints.resolve()
    dataset_root = args.dataset.resolve()
    output_dir = args.output.resolve()
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {output_dir}")
    output_dir.mkdir(parents=True)

    missing = [name for name in MODEL_SPECS if not (checkpoint_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing checkpoints: {missing}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    reference_model, classes = load_model(
        checkpoint_dir / REFERENCE_FILE, MODEL_SPECS[REFERENCE_FILE][0], device
    )
    items = dataset_items(dataset_root, classes)
    selected = select_examples(
        reference_model,
        items,
        device,
        args.scan_limit,
        args.examples_per_group,
        args.seed,
    )
    del reference_model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    results = {path: [] for _, path, _, _ in selected}
    for filename, (architecture, display_name) in MODEL_SPECS.items():
        model, model_classes = load_model(checkpoint_dir / filename, architecture, device)
        if model_classes != classes:
            raise ValueError(f"Class order mismatch in {filename}")
        for _, image_path, truth, _ in selected:
            image = Image.open(image_path).convert("RGB")
            tensor = TRANSFORM(image).unsqueeze(0).to(device)
            heatmap, prediction, confidence = gradcam(model, architecture, tensor)
            results[image_path].append(
                {
                    "checkpoint": filename,
                    "model": display_name,
                    "true_index": truth,
                    "true_class": classes[truth],
                    "predicted_index": prediction,
                    "predicted_class": classes[prediction],
                    "confidence": confidence,
                    "correct": prediction == truth,
                    "heatmap": heatmap,
                }
            )
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    csv_rows = []
    for sample_number, (_, image_path, truth, reference_prediction) in enumerate(selected, 1):
        image = Image.open(image_path).convert("RGB")
        display_image = TF.center_crop(image, [INPUT_SIZE, INPUT_SIZE])
        display_array = np.asarray(display_image, dtype=np.float32) / 255.0
        model_results = results[image_path]
        figure, axes = plt.subplots(2, 4, figsize=(18, 9))
        axes = axes.ravel()
        axes[0].imshow(display_array)
        axes[0].set_title(
            f"Original\nTrue: {short_label(classes[truth])}\n"
            f"Reference: {'correct' if truth == reference_prediction else 'incorrect'}"
        )
        axes[0].axis("off")
        for axis, result in zip(axes[1:], model_results):
            resized = Image.fromarray(np.uint8(result["heatmap"] * 255)).resize(
                (INPUT_SIZE, INPUT_SIZE), Image.Resampling.BILINEAR
            )
            axis.imshow(display_array)
            axis.imshow(np.asarray(resized) / 255.0, cmap="jet", alpha=0.42)
            status = "CORRECT" if result["correct"] else "WRONG"
            axis.set_title(
                f"{result['model']} [{status}]\n"
                f"Pred: {short_label(result['predicted_class'])}\n"
                f"Confidence: {result['confidence']:.1%}"
            )
            axis.axis("off")
            csv_rows.append(
                {
                    "sample": sample_number,
                    "image": str(image_path.relative_to(dataset_root)),
                    **{key: value for key, value in result.items() if key != "heatmap"},
                }
            )
        axes[-1].axis("off")
        figure.suptitle("Grad-CAM model comparison", fontsize=16)
        figure.tight_layout()
        figure.savefig(output_dir / f"sample_{sample_number:02d}.png", dpi=200)
        plt.close(figure)

    fieldnames = list(csv_rows[0].keys())
    with (output_dir / "predictions.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_rows)
    metadata = {
        "device": str(device),
        "seed": args.seed,
        "scan_limit": args.scan_limit,
        "examples_per_group": args.examples_per_group,
        "reference_checkpoint": REFERENCE_FILE,
        "dataset": str(dataset_root),
        "checkpoints": str(checkpoint_dir),
        "samples": len(selected),
    }
    (output_dir / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
