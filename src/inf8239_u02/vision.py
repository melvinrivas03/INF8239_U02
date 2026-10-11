"""Fashion-MNIST: descarga reproducible, auditoría, partición y CNN pequeña (PyTorch)."""
import gzip
import hashlib
import urllib.request
from pathlib import Path

import numpy as np
import torch
from torch import nn
from sklearn.model_selection import train_test_split

BASE_URL = "https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/master/data/fashion/"
FILES = {"x_train": "train-images-idx3-ubyte.gz", "y_train": "train-labels-idx1-ubyte.gz",
         "x_test": "t10k-images-idx3-ubyte.gz", "y_test": "t10k-labels-idx1-ubyte.gz"}
CLASSES = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
           "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]
DATA_DIR = "data/raw/fashion_mnist"
SEED = 42


def download_fashion_mnist(data_dir: str = DATA_DIR) -> Path:
    path = Path(data_dir)
    path.mkdir(parents=True, exist_ok=True)
    for name in FILES.values():
        target = path / name
        if not target.exists():
            urllib.request.urlretrieve(BASE_URL + name, target)
    return path


def _read_idx(file: Path, offset: int) -> np.ndarray:
    with gzip.open(file, "rb") as fh:
        return np.frombuffer(fh.read(), dtype=np.uint8, offset=offset)


def load_fashion_mnist(data_dir: str = DATA_DIR):
    p = Path(data_dir)
    x_train = _read_idx(p / FILES["x_train"], 16).reshape(-1, 28, 28)
    y_train = _read_idx(p / FILES["y_train"], 8)
    x_test = _read_idx(p / FILES["x_test"], 16).reshape(-1, 28, 28)
    y_test = _read_idx(p / FILES["y_test"], 8)
    return x_train, y_train, x_test, y_test


def image_hashes(x: np.ndarray) -> np.ndarray:
    return np.array([hashlib.md5(img.tobytes()).hexdigest() for img in x])


def split_train_val(x, y, val_size: float = 0.10, seed: int = SEED):
    """Validación estratificada tomada solo de entrenamiento; la prueba oficial no se toca."""
    return train_test_split(x, y, test_size=val_size, random_state=seed, stratify=y)


def to_tensor(x: np.ndarray) -> torch.Tensor:
    return torch.tensor(x, dtype=torch.float32).unsqueeze(1) / 255.0


class SmallCNN(nn.Module):
    def __init__(self, n_classes: int = 10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 3), nn.ReLU(), nn.MaxPool2d(2),   # 28 -> 26 -> 13
            nn.Conv2d(16, 32, 3), nn.ReLU(), nn.MaxPool2d(2),  # 13 -> 11 -> 5
        )
        self.classifier = nn.Sequential(nn.Flatten(), nn.Dropout(0.25),
                                        nn.Linear(32 * 5 * 5, 64), nn.ReLU(),
                                        nn.Linear(64, n_classes))

    def forward(self, x):
        return self.classifier(self.features(x))


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def predict(model: nn.Module, x: torch.Tensor, batch: int = 1024) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        return torch.cat([model(x[i:i + batch]) for i in range(0, len(x), batch)]).argmax(1).numpy()
