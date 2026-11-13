"""Held-out evaluation using the IMC 2025 scoring rules."""

from itertools import combinations

import numpy as np
import pandas as pd

from .data import preprocess_labels


def split_validation(labels, seed=42):
    """Hold out 20% of the images, stratified by dataset and scene."""
    groups = list(labels.groupby(["dataset", "scene"]).indices.values())
    exact_sizes = np.array([len(group) / 5 for group in groups])
    sizes = exact_sizes.astype(int)
    remaining = (len(labels) + 4) // 5 - sizes.sum()
    order = np.argsort(-(exact_sizes - sizes), kind="stable")
    sizes[order[:remaining]] += 1
    rng = np.random.default_rng(seed)
    held_out = np.zeros(len(labels), dtype=bool)
    for group, size in zip(groups, sizes):
        held_out[rng.choice(group, size=size, replace=False)] = True
    return labels.iloc[~held_out].copy(), labels.iloc[held_out].copy()


def _centers(poses):
    rotations = np.stack(poses["R"])
    translations = np.stack(poses["t"])
    return -np.einsum("nji,nj->ni", rotations, translations)


def _similarity_errors(source, target, indices):
    """Fit a proper rotation, uniform scale and translation to selected points."""
    x, y = source[indices], target[indices]
    x_mean, y_mean = x.mean(axis=0), y.mean(axis=0)
    x, y = x - x_mean, y - y_mean
    variance = np.sum(x ** 2)
    if variance < 1e-12:
        return np.full(len(source), np.inf)
    u, _, vh = np.linalg.svd(x.T @ y)
    u[:, -1] *= np.linalg.det(u @ vh)
    scale = np.sqrt(np.sum(y ** 2) / variance)
    if scale <= 0:
        return np.full(len(source), np.inf)
    aligned = scale * (source - x_mean) @ (u @ vh) + y_mean
    return np.linalg.norm(aligned - target, axis=1)


def _registered_count(source, target, thresholds):
    """Search camera triplets and refine each model on its inliers."""
    finite = np.isfinite(source).all(axis=1) & np.isfinite(target).all(axis=1)
    source, target = source[finite], target[finite]
    best = np.zeros(len(thresholds), dtype=int)
    if len(source) <= 3:
        return 0.0

    refined_sets = set()
    for triple in combinations(range(len(source)), 3):
        errors = _similarity_errors(source, target, list(triple))
        for threshold in thresholds:
            inliers = tuple(np.flatnonzero(errors < threshold))
            if len(inliers) < 3 or inliers in refined_sets:
                continue
            refined_sets.add(inliers)
            refined = _similarity_errors(source, target, list(inliers))
            counts = (refined[:, None] < thresholds).sum(axis=0)
            best = np.maximum(best, counts)
        if np.all(best == len(source)):
            break
    # The three alignment cameras do not earn credit.
    return np.maximum(best - 3, 0).mean()


def score_validation(labels, predictions, thresholds):
    """Return per-dataset pose mAA, clustering precision and harmonic mean."""
    keys = ["dataset", "image"]
    if set(map(tuple, labels[keys].to_numpy())) != set(map(tuple, predictions[keys].to_numpy())):
        raise ValueError("Predictions must cover exactly the held-out images.")
    predicted = predictions.set_index(keys, verify_integrity=True)
    limits = thresholds.set_index(["dataset", "scene"])["thresholds"]
    truth = preprocess_labels(labels)
    results = []
    for dataset, rows in truth.groupby("dataset"):
        clusters = predicted.loc[dataset].groupby("scene")
        candidates = {name: group for name, group in clusters if name != "outliers"}
        pose_total = pose_possible = cluster_hits = cluster_size = 0
        for scene, group in rows.loc[~rows["is_outlier_image"]].groupby("scene"):
            target = group.set_index("image")
            # IMC 2025 drops the two tightest thresholds.
            cutoffs = np.array(limits.loc[dataset, scene].split(";"), dtype=float)[2:]
            pose_possible += len(target) - 3
            best = (0.0, 0.0, 0, 0)
            for cluster in candidates.values():
                common = target.index.intersection(cluster.index)
                if common.empty:
                    continue
                registered = _registered_count(
                    _centers(cluster.loc[common]), _centers(target.loc[common]), cutoffs,
                )
                association = (registered, len(common) / len(cluster), len(common), len(cluster))
                if association[:2] > best[:2]:
                    best = association
            pose_total += best[0]
            cluster_hits += best[2]
            cluster_size += best[3]
        maa = pose_total / pose_possible if pose_possible > 0 else 0.0
        precision = cluster_hits / cluster_size if cluster_size else 0.0
        score = 2 * maa * precision / (maa + precision) if maa + precision else 0.0
        results.append((dataset, maa, precision, score))
    return pd.DataFrame(results, columns=["Dataset", "Pose mAA", "Clustering precision", "Score"])
