"""Safely reproduce the dataset split created by sampling_dataset.ipynb.

The default mode is a dry run. Image files are copied only when --execute is
provided. Existing output and staging directories are never removed or reused.
"""

from __future__ import annotations

import argparse
import ast
from collections import defaultdict
import json
from pathlib import Path
import random
import re
import shutil
import sys


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--classes", type=int, required=True)
    parser.add_argument("--train-per-class", type=int, default=40)
    parser.add_argument("--validation-per-class", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--reference-notebook",
        type=Path,
        help="Notebook whose saved ImageFolder class mapping must match.",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Copy files and write subset JSON. Without this flag, dry-run only.",
    )
    return parser.parse_args()


def read_json(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"Required JSON not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def image_path(image: dict) -> str:
    for key in ("file_name", "path", "filepath"):
        if image.get(key):
            return image[key]
    raise KeyError(f"Image has no supported path field: {image.get('id')}")


def select_categories(val_data: dict, count: int, seed: int) -> list[int]:
    categories = sorted({ann["category_id"] for ann in val_data["annotations"]})
    if count > len(categories):
        raise ValueError(f"Requested {count} classes; only {len(categories)} exist")
    random.seed(seed)
    return random.sample(categories, count)


def selected_images(data: dict, category_ids: set[int]) -> list[dict]:
    selected_image_ids = {
        ann["image_id"]
        for ann in data["annotations"]
        if ann["category_id"] in category_ids
    }
    return [img for img in data["images"] if img["id"] in selected_image_ids]


def split_training_images(
    images: list[dict], train_count: int, validation_count: int, seed: int
) -> tuple[list[dict], list[dict]]:
    by_folder: dict[Path, list[dict]] = defaultdict(list)
    for image in images:
        by_folder[Path(image_path(image)).parent].append(image)

    rng = random.Random(seed)
    train_images: list[dict] = []
    validation_images: list[dict] = []
    required = train_count + validation_count

    for folder in sorted(by_folder, key=str):
        folder_images = sorted(by_folder[folder], key=lambda item: item["id"])
        if len(folder_images) < required:
            raise ValueError(
                f"{folder} has {len(folder_images)} images; {required} required"
            )
        rng.shuffle(folder_images)
        train_images.extend(folder_images[:train_count])
        validation_images.extend(folder_images[train_count:required])

    return train_images, validation_images


def category_folder_names(data: dict, selected_ids: set[int]) -> list[str]:
    names = [
        category["image_dir_name"]
        for category in data["categories"]
        if category["id"] in selected_ids
    ]
    return sorted(names)


def notebook_class_mapping(path: Path) -> dict[str, int]:
    notebook = read_json(path)
    output_text: list[str] = []
    for cell in notebook.get("cells", []):
        for output in cell.get("outputs", []):
            text = output.get("text")
            if isinstance(text, list):
                output_text.append("".join(text))
            elif isinstance(text, str):
                output_text.append(text)

    combined = "\n".join(output_text)
    match = re.search(r"Class mapping:\s*(\{[^\n]+\})", combined)
    if not match:
        raise ValueError(f"No saved 'Class mapping' output found in {path}")
    mapping = ast.literal_eval(match.group(1))
    if not isinstance(mapping, dict):
        raise TypeError(f"Invalid class mapping in {path}")
    return mapping


def build_subset_json(
    original: dict,
    images: list[dict],
    selected_ids: set[int],
    output_split: str,
) -> dict:
    image_ids = {image["id"] for image in images}
    subset = {
        key: value
        for key, value in original.items()
        if key not in ("images", "annotations", "categories")
    }
    subset["images"] = []
    for image in images:
        copied = image.copy()
        key = next(k for k in ("file_name", "path", "filepath") if copied.get(k))
        old_path = Path(copied[key])
        copied[key] = (Path(output_split) / Path(*old_path.parts[1:])).as_posix()
        subset["images"].append(copied)
    subset["annotations"] = [
        ann for ann in original["annotations"] if ann["image_id"] in image_ids
    ]
    subset["categories"] = [
        category
        for category in original["categories"]
        if category["id"] in selected_ids
    ]
    return subset


def source_and_relative(image: dict, source: Path, source_split: str, dest_split: str):
    original = Path(image_path(image))
    relative = Path(*original.parts[1:])
    return source / source_split / relative, Path(dest_split) / relative


def main() -> int:
    args = parse_args()
    source = args.source.resolve()
    output = args.output.resolve()
    manifest_path = args.manifest.resolve()
    staging = output.with_name(output.name + ".building")

    train_data = read_json(source / "train_mini.json")
    val_data = read_json(source / "val.json")
    selected = select_categories(val_data, args.classes, args.seed)
    selected_ids = set(selected)

    chosen_train_images = selected_images(train_data, selected_ids)
    train_images, validation_images = split_training_images(
        chosen_train_images,
        args.train_per_class,
        args.validation_per_class,
        args.seed,
    )
    test_images = selected_images(val_data, selected_ids)
    folder_names = category_folder_names(val_data, selected_ids)
    expected_mapping = {name: index for index, name in enumerate(folder_names)}

    expected_counts = {
        "classes": args.classes,
        "train_mini": args.classes * args.train_per_class,
        "validation": args.classes * args.validation_per_class,
        "val": args.classes * 10,
    }
    actual_counts = {
        "classes": len(folder_names),
        "train_mini": len(train_images),
        "validation": len(validation_images),
        "val": len(test_images),
    }
    if actual_counts != expected_counts:
        raise ValueError(f"Count mismatch: expected {expected_counts}, got {actual_counts}")

    reference_status = "not requested"
    if args.reference_notebook:
        reference_mapping = notebook_class_mapping(args.reference_notebook.resolve())
        if reference_mapping != expected_mapping:
            missing = sorted(set(reference_mapping) - set(expected_mapping))[:5]
            extra = sorted(set(expected_mapping) - set(reference_mapping))[:5]
            raise ValueError(
                "Reference notebook class mapping differs. "
                f"Missing examples: {missing}; extra examples: {extra}"
            )
        reference_status = f"matched {args.reference_notebook.resolve()}"

    split_specs = [
        (train_images, "train_mini", "train_mini"),
        (validation_images, "train_mini", "validation"),
        (test_images, "val", "val"),
    ]
    files: dict[str, list[dict[str, object]]] = defaultdict(list)
    missing_sources: list[str] = []
    for images, source_split, destination_split in split_specs:
        for image in images:
            src, relative = source_and_relative(
                image, source, source_split, destination_split
            )
            if not src.is_file():
                missing_sources.append(str(src))
            files[destination_split].append(
                {
                    "image_id": image["id"],
                    "source": str(src),
                    "destination": relative.as_posix(),
                }
            )
    if missing_sources:
        raise FileNotFoundError(
            f"{len(missing_sources)} source images missing; first: {missing_sources[0]}"
        )

    manifest = {
        "algorithm_source": "src/data_processing/sampling_dataset.ipynb",
        "source": str(source),
        "output": str(output),
        "parameters": {
            "classes": args.classes,
            "train_per_class": args.train_per_class,
            "validation_per_class": args.validation_per_class,
            "seed": args.seed,
        },
        "counts": actual_counts,
        "reference_mapping": reference_status,
        "selected_category_ids_in_sample_order": selected,
        "class_to_idx": expected_mapping,
        "files": files,
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    if manifest_path.exists():
        existing_manifest = read_json(manifest_path)
        if existing_manifest != manifest:
            raise FileExistsError(
                f"Manifest exists with different content; refusing to overwrite: {manifest_path}"
            )
        manifest_status = "verified existing"
    else:
        with manifest_path.open("x", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2)
        manifest_status = "written"

    print(f"Mode: {'EXECUTE' if args.execute else 'DRY RUN'}")
    print(f"Counts: {actual_counts}")
    print(f"Reference mapping: {reference_status}")
    print(f"Manifest {manifest_status}: {manifest_path}")

    if not args.execute:
        print("No dataset files were copied. Review the manifest, then use --execute.")
        return 0

    if output.exists():
        raise FileExistsError(f"Output already exists; refusing to overwrite: {output}")
    if staging.exists():
        raise FileExistsError(f"Staging directory exists; refusing to reuse: {staging}")
    staging.mkdir(parents=True)

    for destination_split, entries in files.items():
        for index, entry in enumerate(entries, start=1):
            destination = staging / str(entry["destination"])
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(str(entry["source"]), destination)
            if index % 1000 == 0:
                print(f"Copied {destination_split}: {index}/{len(entries)}")

    json_outputs = {
        "train_mini.json": build_subset_json(
            train_data, train_images, selected_ids, "train_mini"
        ),
        "validation.json": build_subset_json(
            train_data, validation_images, selected_ids, "validation"
        ),
        "val.json": build_subset_json(val_data, test_images, selected_ids, "val"),
    }
    for filename, data in json_outputs.items():
        with (staging / filename).open("x", encoding="utf-8") as handle:
            json.dump(data, handle, indent=2)

    staging.rename(output)
    print(f"Dataset completed: {output}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(1)
