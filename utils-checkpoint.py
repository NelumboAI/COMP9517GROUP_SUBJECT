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