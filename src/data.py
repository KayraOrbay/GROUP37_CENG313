"""Read Image Matching Challenge 2025 training labels."""

import csv
from pathlib import Path


def read_train_labels(path):
    """Return label records after checking the required CSV columns."""
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {
            "dataset", "scene", "image", "rotation_matrix", "translation_vector"
        }
        if not required.issubset(reader.fieldnames or []):
            raise ValueError("Beklenen train_labels.csv sütunları eksik.")
        rows = list(reader)
    if not rows:
        raise ValueError("Etiket dosyası boş.")
    return rows
