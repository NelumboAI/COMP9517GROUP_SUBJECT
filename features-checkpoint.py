import cv2
import numpy as np
import pickle
from pathlib import Path
from tqdm import tqdm
from sklearn.cluster import KMeans
from skimage.feature import local_binary_pattern


def extract_hog(image_path, resize=(128, 128)):
    img = cv2.imread(str(image_path))
    if img is None:
        return None
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, resize)

    from skimage.feature import hog
    features = hog(
        gray,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        visualize=False,
        channel_axis=None
    )
    return features.astype(np.float32)

def extract_lbp(image_path, resize=(128, 128), radius=1, n_points=8):
    img = cv2.imread(str(image_path))
    if img is None:
        return None
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, resize)
    lbp = local_binary_pattern(gray, n_points, radius, method='uniform')
    hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, n_points + 3), range=(0, n_points + 2))
    hist = hist.astype(np.float32)
    hist /= (hist.sum() + 1e-7)
    return hist

def extract_sift_descriptors(image_path, resize=(256, 256)):
    img = cv2.imread(str(image_path))
    if img is None:
        return None
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, resize)
    sift = cv2.SIFT_create()
    _, des = sift.detectAndCompute(gray, None)
    return des

def build_sift_vocabulary(image_paths, n_clusters=500, max_images=2000, seed=42):
    all_desc = []
    for path in tqdm(image_paths[:max_images]):
        des = extract_sift_descriptors(path)
        if des is not None and len(des) > 0:
            all_desc.extend(des)
    all_desc = np.array(all_desc)
    kmeans = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10)
    kmeans.fit(all_desc)
    return kmeans

def extract_sift_bovw(image_path, kmeans, resize=(256, 256)):
    des = extract_sift_descriptors(image_path, resize)
    if des is None or len(des) == 0:
        return np.zeros(kmeans.n_clusters)
    words = kmeans.predict(des)
    hist = np.bincount(words, minlength=kmeans.n_clusters)
    hist = hist.astype(np.float32)
    hist /= (hist.sum() + 1e-7)
    return hist

def extract_features_batch(image_paths, feature_type, **kwargs):
    features, valid_paths = [], []
    if feature_type == 'hog':
        resize = kwargs.get('resize', (128, 128))
        func = lambda p: extract_hog(p, resize)
    elif feature_type == 'lbp':
        resize = kwargs.get('resize', (128, 128))
        func = lambda p: extract_lbp(p, resize)
    elif feature_type == 'sift':
        kmeans = kwargs.get('kmeans')
        resize = kwargs.get('resize', (256, 256))
        func = lambda p: extract_sift_bovw(p, kmeans, resize)
    else:
        raise ValueError(f"Unknown feature type: {feature_type}")

    for path in tqdm(image_paths, desc=f"Extracting {feature_type.upper()}"):
        feat = func(path)
        if feat is not None:
            features.append(feat)
            valid_paths.append(path)
    return np.array(features), valid_paths

def save_features(features, labels, paths, feature_type, save_dir):
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    np.save(save_dir / f"X_{feature_type}.npy", features)
    np.save(save_dir / f"y_{feature_type}.npy", np.array(labels))
    with open(save_dir / f"paths_{feature_type}.pkl", 'wb') as f:
        pickle.dump(paths, f)
    return save_dir