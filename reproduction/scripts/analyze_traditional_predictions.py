"""Recompute traditional-method metrics from saved prediction arrays."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)


METHODS = ("HOG_SVM", "HOG_RF", "LBP_SVM", "LBP_RF", "SIFT_SVM", "SIFT_RF")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-predictions", type=Path, required=True)
    parser.add_argument("--reproduced-predictions", type=Path, required=True)
    parser.add_argument("--dataset-json", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def category_names(json_path: Path) -> dict[int, str]:
    with json_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    return {
        int(category["id"]): category.get("name") or category.get("image_dir_name")
        for category in data["categories"]
    }


def load_pair(folder: Path, method: str) -> tuple[np.ndarray, np.ndarray] | None:
    prediction_path = folder / f"{method}_y_pred.npy"
    truth_path = folder / f"{method}_y_true.npy"
    if not prediction_path.is_file() or not truth_path.is_file():
        return None
    return np.load(truth_path), np.load(prediction_path)


def analyse_pair(
    source: str,
    method: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    names: dict[int, str],
    output: Path,
) -> dict[str, object]:
    if y_true.shape != y_pred.shape:
        raise ValueError(f"Shape mismatch for {source}/{method}: {y_true.shape} vs {y_pred.shape}")
    labels = np.array(sorted(set(y_true.tolist()) | set(y_pred.tolist())), dtype=int)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    matrix = confusion_matrix(y_true, y_pred, labels=labels)

    method_dir = output / source / method
    method_dir.mkdir(parents=True, exist_ok=False)
    np.save(method_dir / "confusion_matrix.npy", matrix)
    np.save(method_dir / "labels.npy", labels)

    with (method_dir / "per_class_metrics.csv").open("x", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("category_id", "category_name", "precision", "recall", "f1", "support"))
        for label, p, r, score, count in zip(labels, precision, recall, f1, support):
            writer.writerow((int(label), names.get(int(label), "unknown"), p, r, score, int(count)))

    pairs: list[tuple[int, int, int]] = []
    for row in range(len(labels)):
        for column in range(len(labels)):
            if row != column and matrix[row, column] > 0:
                pairs.append((int(matrix[row, column]), int(labels[row]), int(labels[column])))
    pairs.sort(reverse=True)
    with (method_dir / "hardest_confusions.csv").open("x", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("count", "true_id", "true_name", "predicted_id", "predicted_name"))
        for count, true_id, predicted_id in pairs[:50]:
            writer.writerow(
                (
                    count,
                    true_id,
                    names.get(true_id, "unknown"),
                    predicted_id,
                    names.get(predicted_id, "unknown"),
                )
            )

    return {
        "source": source,
        "method": method,
        "samples": int(len(y_true)),
        "classes_in_truth": int(len(set(y_true.tolist()))),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(np.mean(precision)),
        "macro_recall": float(np.mean(recall)),
        "macro_f1": float(np.mean(f1)),
        "top5_accuracy": None,
        "top5_note": "Unavailable: saved arrays contain only top-1 predictions, not class scores.",
    }


def main() -> int:
    args = parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing analysis: {output}")
    output.mkdir(parents=True, exist_ok=False)
    names = category_names(args.dataset_json.resolve())

    summaries: list[dict[str, object]] = []
    pairs_by_source: dict[str, dict[str, tuple[np.ndarray, np.ndarray]]] = {
        "teammate": {},
        "reproduced": {},
    }
    for source, folder in (
        ("teammate", args.reference_predictions.resolve()),
        ("reproduced", args.reproduced_predictions.resolve()),
    ):
        for method in METHODS:
            pair = load_pair(folder, method)
            if pair is None:
                continue
            pairs_by_source[source][method] = pair
            summaries.append(analyse_pair(source, method, *pair, names, output))

    comparisons: list[dict[str, object]] = []
    common_methods = sorted(
        set(pairs_by_source["teammate"]) & set(pairs_by_source["reproduced"])
    )
    for method in common_methods:
        reference_truth, reference_prediction = pairs_by_source["teammate"][method]
        reproduced_truth, reproduced_prediction = pairs_by_source["reproduced"][method]
        same_truth = bool(np.array_equal(reference_truth, reproduced_truth))
        comparisons.append(
            {
                "method": method,
                "same_ground_truth_order": same_truth,
                "prediction_agreement": (
                    float(np.mean(reference_prediction == reproduced_prediction))
                    if same_truth
                    else None
                ),
            }
        )

    with (output / "metrics.json").open("x", encoding="utf-8") as handle:
        json.dump(summaries, handle, indent=2)
    with (output / "comparisons.json").open("x", encoding="utf-8") as handle:
        json.dump(comparisons, handle, indent=2)
    with (output / "metrics.csv").open("x", newline="", encoding="utf-8") as handle:
        fields = tuple(summaries[0].keys()) if summaries else ()
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(summaries)

    print(json.dumps({"metrics": summaries, "comparisons": comparisons}, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(1)
