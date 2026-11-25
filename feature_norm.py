"""Feature normalization helpers shared by training and evaluation."""
from collections import defaultdict

import numpy as np


def file_mean_map(features, files, min_segments=2):
    """Mean feature vector per recording, or None for short recordings."""
    X = np.asarray(features, dtype=np.float32)
    groups = defaultdict(list)
    for idx, path in enumerate(files):
        groups[path].append(idx)
    return {
        path: (
            X[indices].mean(axis=0)
            if len(indices) >= min_segments
            else None
        )
        for path, indices in groups.items()
    }


def apply_file_means(features, files, means):
    X = np.asarray(features, dtype=np.float32).copy()
    for idx, path in enumerate(files):
        mean = means.get(path)
        if mean is not None:
            X[idx] -= mean
    return X


def apply_owner_means(features, owners, files, means):
    X = np.asarray(features, dtype=np.float32).copy()
    for idx, owner in enumerate(owners):
        mean = means.get(files[owner])
        if mean is not None:
            X[idx] -= mean
    return X


def file_cmvn(features, files, min_segments=2):
    """Subtract each recording's mean feature vector (unsupervised CMVN).

    Recordings with fewer than `min_segments` segments are left unchanged;
    subtracting a file's only segment would erase the sample entirely. At
    inference the same idea is implemented with a rolling session buffer.
    """
    means = file_mean_map(features, files, min_segments)
    X = apply_file_means(features, files, means)
    normalized = 0
    groups = defaultdict(int)
    for path in files:
        groups[path] += 1
    for path, count in groups.items():
        if means.get(path) is not None:
            normalized += count
    return X, normalized
