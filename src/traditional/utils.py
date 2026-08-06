import json
from pathlib import Path

def load_dataset(json_path, base_dir):
    with open(json_path, 'r') as f:
        data = json.load(f)

    id_to_path = {}
    for img in data['images']:
        id_to_path[img['id']] = base_dir / img['file_name']

    id_to_label = {}
    for ann in data['annotations']:
        id_to_label[ann['image_id']] = ann['category_id']

    paths, labels = [], []
    for img_id, path in id_to_path.items():
        if img_id in id_to_label:
            paths.append(path)
            labels.append(id_to_label[img_id])

    return paths, labels

from collections import defaultdict
import numpy as np

def sample_classes_and_split(
    paths,
    labels,
    n_classes=500,
    train_per_class=40,
    val_per_class=10,
    seed=0
):
    rng = np.random.default_rng(seed)

    class_to_paths = defaultdict(list)

    for path, label in zip(paths, labels):
        class_to_paths[label].append(path)

    valid_classes = [
        c for c, items in class_to_paths.items()
        if len(items) >= train_per_class + val_per_class
    ]

    if len(valid_classes) < n_classes:
        raise ValueError(
            f"Not enough valid classes. Required {n_classes}, but only found {len(valid_classes)}."
        )

    selected_classes = rng.choice(
        valid_classes,
        size=n_classes,
        replace=False
    )

    train_paths, train_labels = [], []
    val_paths, val_labels = [], []

    for c in selected_classes:
        items = list(class_to_paths[c])
        rng.shuffle(items)

        train_items = items[:train_per_class]
        val_items = items[train_per_class:train_per_class + val_per_class]

        train_paths.extend(train_items)
        train_labels.extend([c] * len(train_items))

        val_paths.extend(val_items)
        val_labels.extend([c] * len(val_items))

    return train_paths, train_labels, val_paths, val_labels, set(selected_classes)