import yaml
import json
import time
import csv
import pickle
import numpy as np
from pathlib import Path

from sklearn.svm import SVC, LinearSVC
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.traditional.utils import load_dataset, sample_classes_and_split
from src.traditional.features import extract_features_batch, build_sift_vocabulary


def convert_yaml_params(params):
    """Convert YAML list parameters into Python tuple parameters."""
    params = dict(params)

    if "resize" in params:
        params["resize"] = tuple(params["resize"])

    if "pixels_per_cell" in params:
        params["pixels_per_cell"] = tuple(params["pixels_per_cell"])

    if "cells_per_block" in params:
        params["cells_per_block"] = tuple(params["cells_per_block"])

    return params


def build_classifier(classifier_type, params):
    if classifier_type == "svm":
        return SVC(
            kernel=params.get("kernel", "rbf"),
            C=params.get("C", 1.0),
            gamma=params.get("gamma", "scale"),
            random_state=0
        )

    elif classifier_type == "linear_svm":
        return LinearSVC(
            C=params.get("C", 1.0),
            max_iter=params.get("max_iter", 5000),
            random_state=0,
            dual=False
        )
    
    elif classifier_type == "sgd_svm":
        return SGDClassifier(
            loss="hinge",
            alpha=params.get("alpha", 0.0001),
            max_iter=params.get("max_iter", 1000),
            tol=params.get("tol", 1e-3),
            random_state=0,
            n_jobs=-1
    )

    elif classifier_type == "rf":
        return RandomForestClassifier(
            n_estimators=params.get("n_estimators", 100),
            max_depth=params.get("max_depth", None),
            random_state=0,
            n_jobs=-1
        )

    else:
        raise ValueError(f"Unknown classifier type: {classifier_type}")


def evaluate_classifier(clf, X, y_true):
    y_pred = clf.predict(X)

    accuracy = accuracy_score(y_true, y_pred)
    precision_macro = precision_score(y_true, y_pred, average="macro", zero_division=0)
    recall_macro = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)

    top5_accuracy = None

    if hasattr(clf, "predict_proba"):
        scores = clf.predict_proba(X)
        top5_indices = np.argsort(scores, axis=1)[:, -5:]
        top5_labels = clf.classes_[top5_indices]
        top5_accuracy = np.mean([
            y_true[i] in top5_labels[i]
            for i in range(len(y_true))
        ])

    elif hasattr(clf, "decision_function"):
        scores = clf.decision_function(X)
        top5_indices = np.argsort(scores, axis=1)[:, -5:]
        top5_labels = clf.classes_[top5_indices]
        top5_accuracy = np.mean([
            y_true[i] in top5_labels[i]
            for i in range(len(y_true))
        ])

    cm = confusion_matrix(y_true, y_pred, labels=clf.classes_)

    return {
        "accuracy": accuracy,
        "precision_macro": precision_macro,
        "recall_macro": recall_macro,
        "f1_macro": f1_macro,
        "top5_accuracy": top5_accuracy,
        "confusion_matrix": cm,
        "y_pred": y_pred
    }


def train_and_evaluate(
    X_train,
    y_train,
    X_val,
    y_val,
    X_test,
    y_test,
    classifier_type,
    classifier_params
):
    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    clf = build_classifier(classifier_type, classifier_params)

    start_train = time.time()
    clf.fit(X_train_scaled, y_train)
    training_time = time.time() - start_train

    start_eval = time.time()
    val_metrics = evaluate_classifier(clf, X_val_scaled, y_val)
    test_metrics = evaluate_classifier(clf, X_test_scaled, y_test)
    testing_time = time.time() - start_eval

    return clf, scaler, val_metrics, test_metrics, training_time, testing_time


def save_csv(rows, path):
    if not rows:
        return

    fieldnames = list(rows[0].keys())

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main(config_path):
    config_path = Path(config_path)

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    root_dir = Path(config["data"]["root_dir"])

    print("Loading dataset...")

    all_train_paths, all_train_labels = load_dataset(
        root_dir / "train_mini.json",
        root_dir
    )

    all_test_paths, all_test_labels = load_dataset(
        root_dir / "val.json",
        root_dir
    )

    train_paths, train_labels, val_paths, val_labels, selected_classes = sample_classes_and_split(
        all_train_paths,
        all_train_labels,
        n_classes=config["experiment"]["n_classes"],
        train_per_class=config["experiment"]["train_per_class"],
        val_per_class=config["experiment"]["val_per_class"],
        seed=config["experiment"]["seed"]
    )

    test_paths = []
    test_labels = []

    for path, label in zip(all_test_paths, all_test_labels):
        if label in selected_classes:
            test_paths.append(path)
            test_labels.append(label)

    print(f"Selected classes: {len(selected_classes)}")
    print(f"Train images: {len(train_paths)}")
    print(f"Validation images: {len(val_paths)}")
    print(f"Test images: {len(test_paths)}")

    results_dir = Path("results")
    results_dir.mkdir(parents=True, exist_ok=True)

    model_dir = results_dir / "models"
    model_dir.mkdir(parents=True, exist_ok=True)

    pred_dir = results_dir / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)

    ablation_rows = []

    for feature_type, feature_param_list in config["features"].items():
        for feature_params in feature_param_list:

            feature_name = feature_params["name"]
            current_feature_params = dict(feature_params)
            current_feature_params.pop("name")
            current_feature_params = convert_yaml_params(current_feature_params)

            print("\n" + "=" * 60)
            print(f"Feature: {feature_name}")
            print("=" * 60)

            if feature_type == "sift":
                vocab_size = current_feature_params.pop("vocab_size")

                print(f"Building SIFT vocabulary with {vocab_size} clusters...")

                kmeans = build_sift_vocabulary(
                    train_paths,
                    n_clusters=vocab_size,
                    seed=config["experiment"]["seed"]
                )

                current_feature_params["kmeans"] = kmeans

            start_feature_time = time.time()

            X_train, valid_train_paths = extract_features_batch(
                train_paths,
                feature_type,
                **current_feature_params
            )

            X_val, valid_val_paths = extract_features_batch(
                val_paths,
                feature_type,
                **current_feature_params
            )

            X_test, valid_test_paths = extract_features_batch(
                test_paths,
                feature_type,
                **current_feature_params
            )

            feature_extraction_time = time.time() - start_feature_time

            train_label_map = {p: y for p, y in zip(train_paths, train_labels)}
            val_label_map = {p: y for p, y in zip(val_paths, val_labels)}
            test_label_map = {p: y for p, y in zip(test_paths, test_labels)}

            y_train = np.array([train_label_map[p] for p in valid_train_paths])
            y_val = np.array([val_label_map[p] for p in valid_val_paths])
            y_test = np.array([test_label_map[p] for p in valid_test_paths])

            print(f"X_train shape: {X_train.shape}")
            print(f"X_val shape: {X_val.shape}")
            print(f"X_test shape: {X_test.shape}")

            for classifier_type, classifier_param_list in config["classifiers"].items():
                for classifier_params in classifier_param_list:

                    classifier_name = classifier_params["name"]
                    current_classifier_params = dict(classifier_params)
                    current_classifier_params.pop("name")

                    experiment_name = f"{feature_name}_{classifier_name}"

                    print("\n" + "-" * 60)
                    print(f"Training experiment: {experiment_name}")
                    print("-" * 60)

                    clf, scaler, val_metrics, test_metrics, training_time, testing_time = train_and_evaluate(
                        X_train,
                        y_train,
                        X_val,
                        y_val,
                        X_test,
                        y_test,
                        classifier_type,
                        current_classifier_params
                    )

                    row = {
                        "experiment_name": experiment_name,
                        "feature_type": feature_type,
                        "feature_name": feature_name,
                        "classifier_type": classifier_type,
                        "classifier_name": classifier_name,
                        "val_accuracy": val_metrics["accuracy"],
                        "val_precision_macro": val_metrics["precision_macro"],
                        "val_recall_macro": val_metrics["recall_macro"],
                        "val_f1_macro": val_metrics["f1_macro"],
                        "test_accuracy": test_metrics["accuracy"],
                        "test_precision_macro": test_metrics["precision_macro"],
                        "test_recall_macro": test_metrics["recall_macro"],
                        "test_f1_macro": test_metrics["f1_macro"],
                        "test_top5_accuracy": test_metrics["top5_accuracy"],
                        "feature_extraction_time": feature_extraction_time,
                        "training_time": training_time,
                        "testing_time": testing_time
                    }

                    ablation_rows.append(row)

                    with open(model_dir / f"{experiment_name}.pkl", "wb") as f:
                        pickle.dump(
                            {
                                "classifier": clf,
                                "scaler": scaler
                            },
                            f
                        )

                    np.save(pred_dir / f"{experiment_name}_y_true.npy", y_test)
                    np.save(pred_dir / f"{experiment_name}_y_pred.npy", test_metrics["y_pred"])
                    np.save(pred_dir / f"{experiment_name}_confusion_matrix.npy", test_metrics["confusion_matrix"])

                    save_csv(ablation_rows, results_dir / "ablation_results.csv")

                    with open(results_dir / "ablation_results.json", "w") as f:
                        json.dump(ablation_rows, f, indent=2)

                    print(f"Validation accuracy: {val_metrics['accuracy']:.4f}")
                    print(f"Validation macro-F1: {val_metrics['f1_macro']:.4f}")
                    print(f"Test accuracy: {test_metrics['accuracy']:.4f}")
                    print(f"Test macro-F1: {test_metrics['f1_macro']:.4f}")
                    print(f"Test top-5 accuracy: {test_metrics['top5_accuracy']}")

    print("\nAblation study finished.")
    print(f"Results saved to {results_dir / 'ablation_results.csv'}")


if __name__ == "__main__":
    config_path = sys.argv[1] if len(sys.argv) > 1 else "configs/traditional_ablation.yaml"
    main(config_path)