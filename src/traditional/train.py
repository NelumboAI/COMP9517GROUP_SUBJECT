import yaml
import numpy as np
import pickle
import json
from pathlib import Path
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score

import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.traditional.utils import load_dataset
from src.traditional.features import (
    extract_features_batch,
    build_sift_vocabulary,
    save_features
)

def train_and_evaluate(X_train, y_train, X_test, y_test, classifier_type, params):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    if classifier_type == 'svm':
        clf = SVC(
            kernel=params.get('kernel', 'rbf'),
            C=params.get('C', 1.0),
            gamma=params.get('gamma', 'scale'),
            random_state=42
        )
    elif classifier_type == 'rf':
        clf = RandomForestClassifier(
            n_estimators=params.get('n_estimators', 100),
            max_depth=params.get('max_depth', None),
            random_state=42,
            n_jobs=-1
        )
    else:
        raise ValueError(f"Unknown classifier: {classifier_type}")

    clf.fit(X_train_scaled, y_train)
    y_pred = clf.predict(X_test_scaled)

    acc = accuracy_score(y_test, y_pred)
    f1_macro = f1_score(y_test, y_pred, average='macro')

    top5 = None
    if classifier_type == 'svm' and hasattr(clf, "decision_function"):
        scores = clf.decision_function(X_test_scaled)
        top5_pred = np.argsort(scores, axis=1)[:, -5:]
        top5 = np.mean([y_test[i] in top5_pred[i] for i in range(len(y_test))])

    return {
        'accuracy': acc,
        'f1_macro': f1_macro,
        'top5_accuracy': top5,
        'y_pred': y_pred,
        'classifier': clf,
        'scaler': scaler
    }

def main(config_path):
    config_path = Path(__file__).parent.parent.parent / config_path
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    root_dir = Path(config['data']['root_dir'])

    train_paths, train_labels = load_dataset(root_dir / 'train_mini.json', root_dir)
    test_paths, test_labels = load_dataset(root_dir / 'val.json', root_dir)

    print(f"Train: {len(train_paths)} images, {len(set(train_labels))} classes")
    print(f"Test: {len(test_paths)} images, {len(set(test_labels))} classes")

    feat_dir = Path("results/features")
    feat_dir.mkdir(parents=True, exist_ok=True)

    feature_types = ['hog', 'lbp', 'sift']
    results_summary = {}

    for feat_type in feature_types:
        print(f"\n===== Processing {feat_type.upper()} =====")

        cache_X = feat_dir / f"X_{feat_type}.npy"

        if cache_X.exists():
            print(f"Loading cached {feat_type} features...")
            X_train = np.load(cache_X)
            y_train = np.load(feat_dir / f"y_{feat_type}.npy")

            cache_X_test = feat_dir / f"X_test_{feat_type}.npy"
            if cache_X_test.exists():
                X_test = np.load(cache_X_test)
                y_test = np.load(feat_dir / f"y_test_{feat_type}.npy")
            else:
                if feat_type == 'sift':
                    with open(feat_dir / "sift_vocab.pkl", 'rb') as f:
                        kmeans = pickle.load(f)
                    X_test, test_paths_valid = extract_features_batch(
                        test_paths, 'sift', kmeans=kmeans, resize=config['features']['sift_resize']
                    )
                else:
                    X_test, test_paths_valid = extract_features_batch(
                        test_paths, feat_type, resize=config['features'][f'{feat_type}_resize']
                    )
                path_to_label = {p: l for p, l in zip(test_paths, test_labels)}
                y_test = np.array([path_to_label[p] for p in test_paths_valid])
                np.save(feat_dir / f"X_test_{feat_type}.npy", X_test)
                np.save(feat_dir / f"y_test_{feat_type}.npy", y_test)
        else:
            if feat_type == 'sift':
                vocab = build_sift_vocabulary(
                    train_paths,
                    n_clusters=config['features']['sift_vocab_size'],
                    seed=config['experiment']['seed']
                )
                with open(feat_dir / "sift_vocab.pkl", 'wb') as f:
                    pickle.dump(vocab, f)
                X_train, train_paths_valid = extract_features_batch(
                    train_paths, 'sift', kmeans=vocab, resize=config['features']['sift_resize']
                )
                X_test, test_paths_valid = extract_features_batch(
                    test_paths, 'sift', kmeans=vocab, resize=config['features']['sift_resize']
                )
            else:
                X_train, train_paths_valid = extract_features_batch(
                    train_paths, feat_type, resize=config['features'][f'{feat_type}_resize']
                )
                X_test, test_paths_valid = extract_features_batch(
                    test_paths, feat_type, resize=config['features'][f'{feat_type}_resize']
                )

            path_to_label_train = {p: l for p, l in zip(train_paths, train_labels)}
            y_train = np.array([path_to_label_train[p] for p in train_paths_valid])

            path_to_label_test = {p: l for p, l in zip(test_paths, test_labels)}
            y_test = np.array([path_to_label_test[p] for p in test_paths_valid])

            save_features(X_train, y_train, train_paths_valid, feat_type, feat_dir)
            np.save(feat_dir / f"X_test_{feat_type}.npy", X_test)
            np.save(feat_dir / f"y_test_{feat_type}.npy", y_test)

        print(f"X_train shape: {X_train.shape}, X_test shape: {X_test.shape}")

        for clf_name in ['svm', 'rf']:
            print(f"\nTraining {clf_name.upper()} on {feat_type.upper()}...")
            params = config['classifier'][clf_name]
            result = train_and_evaluate(X_train, y_train, X_test, y_test, clf_name, params)

            key = f"{feat_type.upper()}_{clf_name.upper()}"
            results_summary[key] = {
                'accuracy': result['accuracy'],
                'f1_macro': result['f1_macro'],
                'top5_accuracy': result['top5_accuracy']
            }

            model_path = Path("results/models") / f"{key}.pkl"
            model_path.parent.mkdir(parents=True, exist_ok=True)
            with open(model_path, 'wb') as f:
                pickle.dump(result['classifier'], f)

            pred_path = Path("results/predictions") / f"{key}_y_pred.npy"
            pred_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(pred_path, result['y_pred'])
            np.save(Path("results/predictions") / f"{key}_y_true.npy", y_test)

    with open("results/summary.json", 'w') as f:
        json.dump(results_summary, f, indent=2)

    print("\n===== All done! Results saved to results/summary.json =====")
    print(results_summary)

if __name__ == "__main__":
    import sys
    config_path = sys.argv[1] if len(sys.argv) > 1 else "configs/baseline.yaml"
    main(config_path)