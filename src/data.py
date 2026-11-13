"""Read Image Matching Challenge 2025 training labels."""

import csv
from pathlib import Path

import numpy as np
import pandas as pd


def read_train_labels(path):
    """Return label records after checking the required CSV columns."""
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {
            "dataset", "scene", "image", "rotation_matrix", "translation_vector"
        }
        if not required.issubset(reader.fieldnames or []):
            raise ValueError("train_labels.csv is missing required columns.")
        rows = list(reader)
    if not rows:
        raise ValueError("The labels file is empty.")
    return rows


REQUIRED_COLUMNS = ["dataset", "scene", "image", "rotation_matrix", "translation_vector"]

def load_train_labels(data_dir="data"):
    df = pd.read_csv(Path(data_dir) / "train_labels.csv")
    missing = set(REQUIRED_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"train_labels.csv is missing columns: {missing}")
    return df

def load_train_thresholds(data_dir="data"):
    return pd.read_csv(Path(data_dir) / "train_thresholds.csv")

def load_sample_submission(data_dir="data"):
    return pd.read_csv(Path(data_dir) / "sample_submission.csv")


def load_data(data_dir="data"):
    """Load the three competition tables used by the notebook."""
    return (load_train_labels(data_dir), load_train_thresholds(data_dir),
            load_sample_submission(data_dir))

def _parse_vector(value, expected_len):
    if pd.isna(value):
        return np.full(expected_len, np.nan)
    parts = [p for p in str(value).split(";") if p != ""]
    arr = np.array([float(p) for p in parts], dtype=float)
    if arr.size != expected_len:
        raise ValueError(f"Expected {expected_len} values, got {arr.size}: {value!r}")
    return arr

def parse_poses(df):
    out = df.copy()
    R = np.vstack(out["rotation_matrix"].apply(_parse_vector, expected_len=9).to_numpy())
    t = np.vstack(out["translation_vector"].apply(_parse_vector, expected_len=3).to_numpy())
    out[[f"R{i}{j}" for i in range(3) for j in range(3)]] = R
    out[["tx", "ty", "tz"]] = t
    return out


def add_pose_features(df):
    """Parse poses and add simple numeric features for EDA.

    rot_angle_deg : rotation angle of R (0 = same orientation as the reference)
    t_norm        : length of the translation vector
    cx, cy, cz    : camera center in world coordinates, C = -R^T t
    dist_to_center: distance of the camera to the median camera center of its scene
    det_R, orth_err: sanity checks (a valid rotation has det = 1 and R R^T = I)
    """
    out = parse_poses(df)
    r_cols = [f"R{i}{j}" for i in range(3) for j in range(3)]
    R = out[r_cols].to_numpy().reshape(-1, 3, 3)
    t = out[["tx", "ty", "tz"]].to_numpy()

    trace = np.trace(R, axis1=1, axis2=2)
    out["rot_angle_deg"] = np.degrees(np.arccos(np.clip((trace - 1) / 2, -1, 1)))
    out["t_norm"] = np.linalg.norm(t, axis=1)
    C = -np.einsum("nji,nj->ni", R, t)
    out[["cx", "cy", "cz"]] = C
    with np.errstate(invalid="ignore"):  # outlier rows are NaN
        out["det_R"] = np.linalg.det(R)
        out["orth_err"] = np.linalg.norm(R @ R.transpose(0, 2, 1) - np.eye(3), axis=(1, 2))

    centers = out.groupby(["dataset", "scene"])[["cx", "cy", "cz"]].transform("median")
    out["dist_to_center"] = np.linalg.norm(out[["cx", "cy", "cz"]].to_numpy() - centers.to_numpy(), axis=1)
    out["is_outlier_image"] = out["scene"].eq("outliers") | out["tx"].isna()
    return out


def preprocess_labels(df):
    """Clean train_labels for modelling.

    - parse pose strings into numeric 3x3 rotation matrices and length-3 translation vectors
    - flag outlier images (no pose) instead of dropping or imputing them
    - encode scenes as integer cluster ids inside each dataset (outliers = -1)
    - add a unique key, because image names are not unique across datasets
    """
    out = parse_poses(df)
    r_cols = [f"R{i}{j}" for i in range(3) for j in range(3)]
    out["R"] = list(out[r_cols].to_numpy().reshape(-1, 3, 3))
    out["t"] = list(out[["tx", "ty", "tz"]].to_numpy())
    out["is_outlier_image"] = out["scene"].eq("outliers") | out["tx"].isna()

    out["scene_id"] = -1
    for ds, idx in out[~out["is_outlier_image"]].groupby("dataset").groups.items():
        out.loc[idx, "scene_id"] = pd.factorize(out.loc[idx, "scene"], sort=True)[0]
    out["key"] = out["dataset"] + "/" + out["image"]
    return out[["key", "dataset", "scene", "scene_id", "image", "is_outlier_image", "R", "t"]]


def load_image(path, max_size=1024, gray=True):
    """Read an image, convert to grayscale and resize so the longest side is max_size.

    Returns the image and the scale factor (needed to map keypoints back
    to the original resolution).
    """
    import cv2
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE if gray else cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(path)
    h, w = img.shape[:2]
    scale = min(1.0, max_size / max(h, w))
    if scale < 1.0:
        img = cv2.resize(img, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)
    return img, scale
