import json
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import confusion_matrix

import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.traditional.utils import load_dataset, sample_classes_and_split


BEST_EXP = "sift_vocab200_sgd_svm_c1"


def load_category_names(json_path):
    with open(json_path, "r") as f:
        data = json.load(f)

    id_to_name = {}

    for cat in data["categories"]:
        cat_id = cat["id"]

        if "name" in cat:
            id_to_name[cat_id] = cat["name"]
        elif "common_name" in cat:
            id_to_name[cat_id] = cat["common_name"]
        else:
            id_to_name[cat_id] = str(cat_id)

    return id_to_name


def reconstruct_test_paths(config_root="data/inat2021", n_classes=500, seed=42):
    root_dir = Path(config_root)

    all_train_paths, all_train_labels = load_dataset(
        root_dir / "train_mini.json",
        root_dir
    )

    all_test_paths, all_test_labels = load_dataset(
        root_dir / "val.json",
        root_dir
    )

    _, _, _, _, selected_classes = sample_classes_and_split(
        all_train_paths,
        all_train_labels,
        n_classes=n_classes,
        train_per_class=40,
        val_per_class=10,
        seed=seed
    )

    test_paths = []
    test_labels = []

    for path, label in zip(all_test_paths, all_test_labels):
        if label in selected_classes:
            test_paths.append(path)
            test_labels.append(label)

    return test_paths, np.array(test_labels)


def main():
    results_dir = Path("results")
    pred_dir = results_dir / "predictions"
    analysis_dir = results_dir / "error_analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    y_true = np.load(pred_dir / f"{BEST_EXP}_y_true.npy")
    y_pred = np.load(pred_dir / f"{BEST_EXP}_y_pred.npy")

    test_paths, reconstructed_labels = reconstruct_test_paths(
        config_root="data/inat2021",
        n_classes=500,
        seed=0
    )

    test_paths = np.array(test_paths)

    if len(test_paths) != len(y_true):
        print("Warning: reconstructed test paths length does not match y_true length.")
        print("len(test_paths):", len(test_paths))
        print("len(y_true):", len(y_true))

    if not np.array_equal(y_true, reconstructed_labels[:len(y_true)]):
        print("Warning: reconstructed labels do not exactly match saved y_true.")
        print("Image paths may not align perfectly with predictions.")
    else:
        print("Reconstructed test paths match y_true labels.")

    id_to_name = load_category_names("data/inat2021/val.json")

    labels = np.unique(y_true)
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    # ---------- 1. Save full confusion matrix ----------
    cm_df = pd.DataFrame(
        cm,
        index=[id_to_name.get(int(x), str(x)) for x in labels],
        columns=[id_to_name.get(int(x), str(x)) for x in labels]
    )
    cm_df.to_csv(analysis_dir / f"{BEST_EXP}_full_confusion_matrix.csv")

    # ---------- 2. Find most confused class pairs ----------
    confused_pairs = []

    for i, true_label in enumerate(labels):
        for j, pred_label in enumerate(labels):
            if i != j and cm[i, j] > 0:
                confused_pairs.append({
                    "true_label_id": int(true_label),
                    "true_label_name": id_to_name.get(int(true_label), str(true_label)),
                    "pred_label_id": int(pred_label),
                    "pred_label_name": id_to_name.get(int(pred_label), str(pred_label)),
                    "count": int(cm[i, j])
                })

    confused_df = pd.DataFrame(confused_pairs)
    confused_df = confused_df.sort_values("count", ascending=False)
    confused_df.to_csv(analysis_dir / f"{BEST_EXP}_top_confused_pairs.csv", index=False)

    print("\nTop 10 confused pairs:")
    print(confused_df.head(10))

    # ---------- 3. Plot confusion matrix for top confused classes ----------
    top_pairs = confused_df.head(10)

    selected_label_ids = set()
    for _, row in top_pairs.iterrows():
        selected_label_ids.add(row["true_label_id"])
        selected_label_ids.add(row["pred_label_id"])

    selected_label_ids = list(selected_label_ids)
    selected_indices = [np.where(labels == x)[0][0] for x in selected_label_ids if x in labels]

    small_cm = cm[np.ix_(selected_indices, selected_indices)]
    small_labels = [id_to_name.get(int(labels[i]), str(labels[i])) for i in selected_indices]

    plt.figure(figsize=(12, 10))
    plt.imshow(small_cm, interpolation="nearest")
    plt.title(f"Confusion Matrix for Most Confused Classes: {BEST_EXP}")
    plt.xlabel("Predicted class")
    plt.ylabel("True class")
    plt.colorbar()

    plt.xticks(range(len(small_labels)), small_labels, rotation=90, fontsize=7)
    plt.yticks(range(len(small_labels)), small_labels, fontsize=7)

    plt.tight_layout()
    plt.savefig(analysis_dir / f"{BEST_EXP}_confusion_matrix_top_classes.png", dpi=300)
    plt.close()

    # ---------- 4. Save correct and incorrect examples ----------
    correct_indices = np.where(y_true == y_pred)[0]
    wrong_indices = np.where(y_true != y_pred)[0]

    correct_examples = []
    wrong_examples = []

    for idx in correct_indices[:30]:
        correct_examples.append({
            "image_path": str(test_paths[idx]),
            "true_label_id": int(y_true[idx]),
            "true_label_name": id_to_name.get(int(y_true[idx]), str(y_true[idx])),
            "pred_label_id": int(y_pred[idx]),
            "pred_label_name": id_to_name.get(int(y_pred[idx]), str(y_pred[idx]))
        })

    for idx in wrong_indices[:50]:
        wrong_examples.append({
            "image_path": str(test_paths[idx]),
            "true_label_id": int(y_true[idx]),
            "true_label_name": id_to_name.get(int(y_true[idx]), str(y_true[idx])),
            "pred_label_id": int(y_pred[idx]),
            "pred_label_name": id_to_name.get(int(y_pred[idx]), str(y_pred[idx]))
        })

    pd.DataFrame(correct_examples).to_csv(
        analysis_dir / f"{BEST_EXP}_correct_examples.csv",
        index=False
    )

    pd.DataFrame(wrong_examples).to_csv(
        analysis_dir / f"{BEST_EXP}_wrong_examples.csv",
        index=False
    )

    print("\nSaved files to:", analysis_dir)


if __name__ == "__main__":
    main()