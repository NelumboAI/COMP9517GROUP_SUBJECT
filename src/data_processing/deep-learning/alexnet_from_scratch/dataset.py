from torchvision import datasets, transforms
from pathlib import Path
import json
import random
from collections import defaultdict
import shutil

folder = "C:\\dataset\\"
BASE_DIR = Path(folder)
DATA_DIR = BASE_DIR / "data"

training_transform = None

def load_datasets(data_root):
    train_ds = datasets.ImageFolder(f"{DATA_DIR}/train_mini", transform=training_transform)