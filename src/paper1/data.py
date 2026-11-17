from __future__ import annotations
import os
from pathlib import Path
import numpy as np

CLASS_NAMES = ("non_nodule", "nodule")
VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
EXPECTED_COUNTS = {
    "train_non_nodule": 4805, "train_nodule": 5016,
    "valid_non_nodule": 948, "valid_nodule": 919,
    "test_non_nodule": 720, "test_nodule": 766,
}

def dataset_root(path: str | None = None) -> Path:
    value = path or os.environ.get("PAPER1_DATASET_DIR")
    if not value:
        raise EnvironmentError("Set PAPER1_DATASET_DIR or pass --dataset-dir.")
    root = Path(value).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Dataset directory does not exist: {root}")
    return root

def collect_paths(split_dir: str | Path):
    split_dir = Path(split_dir)
    paths, labels = [], []
    for label, class_name in enumerate(CLASS_NAMES):
        class_dir = split_dir / class_name
        if not class_dir.is_dir():
            raise FileNotFoundError(f"Missing class directory: {class_dir}")
        for path in sorted(class_dir.iterdir()):
            if path.is_file() and path.suffix.lower() in VALID_EXTENSIONS:
                paths.append(str(path)); labels.append(label)
    return np.asarray(paths), np.asarray(labels, dtype=np.int32)

def verify_counts(root: str | Path, expected: dict | None = None):
    root = Path(root); expected = expected or EXPECTED_COUNTS; actual = {}
    for split in ("train", "valid", "test"):
        _, labels = collect_paths(root / split)
        actual[f"{split}_non_nodule"] = int(np.sum(labels == 0))
        actual[f"{split}_nodule"] = int(np.sum(labels == 1))
    if actual != expected:
        raise ValueError(f"Dataset counts differ. Expected={expected}; actual={actual}")
    return actual
