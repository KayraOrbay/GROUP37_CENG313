"""A feature-matching baseline for relative camera pose."""

from dataclasses import dataclass
from heapq import heappop, heappush
from itertools import combinations

import cv2
import numpy as np
import pandas as pd

from .data import load_image


@dataclass(frozen=True)
class ImageFeatures:
    points: np.ndarray
    descriptors: np.ndarray | None
    focal_length: float


@dataclass(frozen=True)
class RelativePose:
    rotation: np.ndarray
    translation_direction: np.ndarray
    matches: int
    inliers: int


def extract_features(path):
    """Detect SIFT features and normalize points using approximate intrinsics."""
    image, _ = load_image(path)
    height, width = image.shape
    focal_length = float(max(height, width))
    keypoints, descriptors = cv2.SIFT_create(nfeatures=3000).detectAndCompute(image, None)
    pixels = np.array([point.pt for point in keypoints], dtype=np.float64).reshape(-1, 2)
    center = np.array([(width - 1) / 2, (height - 1) / 2])
    return ImageFeatures((pixels - center) / focal_length, descriptors, focal_length)


def estimate_relative_pose(first, second):
    """Estimate the transform from the first camera to the second, up to scale."""
    if first.descriptors is None or second.descriptors is None:
        return None

    matcher = cv2.BFMatcher(cv2.NORM_L2)
    neighbors = matcher.knnMatch(first.descriptors, second.descriptors, k=2)
    matches = [pair[0] for pair in neighbors
               if len(pair) == 2 and pair[0].distance < 0.75 * pair[1].distance]
    # Require enough matches to estimate a pose robustly above the five-point minimum.
    if len(matches) < 8:
        return None

    points1 = first.points[[match.queryIdx for match in matches]]
    points2 = second.points[[match.trainIdx for match in matches]]
    camera_matrix = np.eye(3)
    essential, mask = cv2.findEssentialMat(
        points1, points2, camera_matrix, method=cv2.RANSAC,
        prob=0.999, threshold=1.5 / min(first.focal_length, second.focal_length),
    )
    if essential is None or essential.shape != (3, 3):
        return None

    inliers, rotation, translation, _ = cv2.recoverPose(
        essential, points1, points2, camera_matrix, mask=mask,
    )
    if inliers < 8:
        return None
    return RelativePose(rotation, translation.ravel(), len(matches), inliers)


def predict_dataset(image_names, image_dir):
    """Cluster a dataset by verified matches and chain relative poses."""
    names = sorted(image_names)
    features = {name: extract_features(image_dir / name) for name in names}
    edges = {name: [] for name in names}
    for first, second in combinations(names, 2):
        pose = estimate_relative_pose(features[first], features[second])
        if pose is None or pose.inliers < 15 or pose.inliers / pose.matches < 0.25:
            continue
        rotation, direction = pose.rotation, pose.translation_direction
        edges[first].append((second, pose.inliers, rotation, direction))
        edges[second].append((first, pose.inliers, rotation.T, -rotation.T @ direction))

    predictions = {}
    cluster = 0
    for root in names:
        if root in predictions:
            continue
        if not edges[root]:
            predictions[root] = ("outliers", np.full((3, 3), np.nan), np.full(3, np.nan))
            continue

        cluster += 1
        scene = f"cluster{cluster}"
        predictions[root] = (scene, np.eye(3), np.zeros(3))
        frontier = []

        def enqueue(source):
            for target, inliers, rotation, direction in edges[source]:
                if target not in predictions:
                    heappush(frontier, (-inliers, source, target, rotation, direction))

        enqueue(root)
        while frontier:
            _, source, target, rotation, direction = heappop(frontier)
            if target in predictions:
                continue
            _, parent_rotation, parent_translation = predictions[source]
            predictions[target] = (
                scene, rotation @ parent_rotation, rotation @ parent_translation + direction,
            )
            enqueue(target)

    return pd.DataFrame.from_dict(
        predictions, orient="index", columns=["scene", "R", "t"],
    ).rename_axis("image").reset_index()


def predict_submission(template, image_root):
    """Predict scenes and poses in the order and format of a Kaggle template."""
    predicted = [
        predict_dataset(group["image"], image_root / dataset).assign(dataset=dataset)
        for dataset, group in template.groupby("dataset")
    ]
    keys = ["dataset", "image"]
    result = template.drop(
        columns=["scene", "rotation_matrix", "translation_vector"],
    ).merge(pd.concat(predicted), on=keys, how="left", validate="one_to_one")
    result["rotation_matrix"] = result["R"].map(_format_pose)
    result["translation_vector"] = result["t"].map(_format_pose)
    return result.loc[:, template.columns]


def _format_pose(array):
    return ";".join(f"{value:.12g}" for value in array.ravel())
