from torchvision import datasets, transforms
from pathlib import Path
import sys

folder = "C:\\dataset\\"
BASE_DIR = Path(folder)
DATA_DIR = BASE_DIR / "sampled_500"

IMG_SIZE = 224

MEAN = [0.485, 0.456, 0.406]
STD_DEV = [0.229, 0.224, 0.225]

training_transform = transforms.Compose([
    transforms.RandomResizedCrop(IMG_SIZE, scale=(0.7, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=MEAN, std=STD_DEV)
])

eval_transform = transforms.Compose([
    transforms.Resize(int(IMG_SIZE * 1.14)),
    transforms.CenterCrop(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=MEAN, std=STD_DEV)
])

def load_datasets(data_root):
    train_ds = datasets.ImageFolder(f"{DATA_DIR}/train_mini", transform=training_transform)
    validation_ds = datasets.ImageFolder(f"{DATA_DIR}/validation", transform=eval_transform)
    test_ds = datasets.ImageFolder(f"{DATA_DIR}/val", transform=eval_transform)
    return train_ds, validation_ds, test_ds

if __name__ == "__main__":
    train_ds, validation_ds, test_ds = load_datasets(DATA_DIR)
    print(f"classes found: {len(train_ds.classes)}")
    print(f"train images:  {len(train_ds)}")
    print(f"val images:    {len(validation_ds)}")
    print(f"test images:   {len(test_ds)}")
 
    # Pull one sample through the pipeline and check the tensor shape --
    # this should print torch.Size([3, 224, 224]) if everything is wired up right.
    image, label = train_ds[0]
    print(f"sample tensor shape: {image.shape}")
    print(f"sample label index:  {label} -> class '{train_ds.classes[label]}'")
