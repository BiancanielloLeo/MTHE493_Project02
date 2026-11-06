"""
data.py - images
  Part 1 is splitting the images, part 2 is loading, where make_loaders() 
  turns the images in split.json into batches of numbers
"""

import json
import random
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

REPO = Path(__file__).parent.parent                  # MTHE493 folder
IMAGES = REPO / "images" / "Oil Spills"
SPLIT_FILE = Path(__file__).parent / "split.json"


# ---------------- Part 2: loading ----------------

# Image transformations before the network sees it.
TRANSFORM = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),     # copy the gray channel 3 times as pretrained ResNet expects 3
    transforms.Resize((224, 224)),                   # 400x400 to 224x224, the size ResNet was built for
    transforms.ToTensor(),                           # picture to grid of numbers between 0 and 1
    transforms.Normalize([0.485, 0.456, 0.406],      # scale the numbers the same way ResNet's
                         [0.229, 0.224, 0.225]),     #   original training images were scaled
])


class OilDataset(Dataset):
    """Given a list of [path, label] pairs, returns image number i as (numbers, label)."""

    def __init__(self, items):
        self.items = items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        path, label = self.items[i]
        image = TRANSFORM(Image.open(REPO / path))
        return image, torch.tensor(label, dtype=torch.float32)


def make_loaders(batch_size=32):
    """Read split.json and return train / val / test loaders that take images in batches."""
    split = json.loads(SPLIT_FILE.read_text())
    train = DataLoader(OilDataset(split["train"]), batch_size=batch_size, shuffle=True)  # new order each round
    val = DataLoader(OilDataset(split["val"]), batch_size=batch_size)
    test = DataLoader(OilDataset(split["test"]), batch_size=batch_size)
    return train, val, test


# ---------------- Part 1: split  ----------------

if __name__ == "__main__":
    random.seed(0)                                    # same shuffle every run
    split = {"train": [], "val": [], "test": []}

    # Split each class separately so every group is approximately 1/3 oil.
    for folder, label in [("Oil", 1), ("No Oil", 0)]:
        paths = sorted((IMAGES / folder).glob("*.jpg"))
        random.shuffle(paths)

        n_train = int(0.70 * len(paths))
        n_val = int(0.15 * len(paths))

        for i, path in enumerate(paths):
            group = "train" if i < n_train else "val" if i < n_train + n_val else "test"
            split[group].append([path.relative_to(REPO).as_posix(), label])   # (image, label) pair

    for group in split.values():                      # mix oil and no-oil together
        random.shuffle(group)

    for name, items in split.items():
        n_oil = sum(label for _, label in items)
        print(f"{name}: {len(items)} images, {n_oil} oil")

    SPLIT_FILE.write_text(json.dumps(split, indent=1))
