"""Out-of-distribution gate for audio features.

The LOCO evaluation showed the classifier can be confidently wrong on a new
corpus. This detector does not try to classify; it answers one question:
"is this feature vector close to any class seen during training?"

Score = minimum per-class Mahalanobis distance in standardized feature space.
The threshold is calibrated on unseen recordings (GroupKFold by source file),
not on the training set itself.
"""
import numpy as np
from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler


class OODDetector:
    def __init__(self, calibration_percentile=85.0, shrinkage=0.01):
        self.calibration_percentile = float(calibration_percentile)
        self.shrinkage = float(shrinkage)
        self.scaler = None
        self.class_means = {}
        self.class_inv_covs = {}
        self.threshold = None
        self.calibration_scores_ = None
        self.metadata = {}

    def _fit_stats(self, X, y):
        scaler = StandardScaler().fit(X)
        Z = scaler.transform(X)
        means, inv_covs = {}, {}
        for cls in np.unique(y):
            block = Z[y == cls]
            if len(block) < 2:
                continue
            means[cls] = block.mean(axis=0)
            cov = np.cov(block, rowvar=False)
            cov = np.atleast_2d(cov) + self.shrinkage * np.eye(Z.shape[1])
            inv_covs[cls] = np.linalg.pinv(cov)
        return scaler, means, inv_covs

    def _score_with(self, X, scaler, means, inv_covs):
        Z = scaler.transform(np.atleast_2d(X))
        distances = []
        for cls, mean in means.items():
            diff = Z - mean
            dist = np.sqrt(np.einsum(
                "ij,jk,ik->i", diff, inv_covs[cls], diff
            ))
            distances.append(dist)
        if not distances:
            return np.full(len(Z), np.inf)
        return np.min(np.vstack(distances), axis=0)

    def fit(self, X, y, groups=None):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y)
        if groups is not None:
            groups = np.asarray(groups)
            splitter = GroupKFold(n_splits=min(5, len(np.unique(groups))))
            splits = list(splitter.split(X, y, groups))
        else:
            min_count = min(np.bincount(y)[np.bincount(y) > 0])
            splitter = StratifiedKFold(
                n_splits=max(2, min(5, int(min_count))),
                shuffle=True,
                random_state=42,
            )
            splits = list(splitter.split(X, y))

        calibration = []
        for train_idx, test_idx in splits:
            scaler, means, inv_covs = self._fit_stats(X[train_idx], y[train_idx])
            calibration.append(
                self._score_with(X[test_idx], scaler, means, inv_covs)
            )
        self.calibration_scores_ = np.concatenate(calibration)
        self.threshold = float(np.percentile(
            self.calibration_scores_, self.calibration_percentile
        ))
        self.scaler, self.class_means, self.class_inv_covs = self._fit_stats(X, y)
        self.metadata = {
            "method": "per_class_mahalanobis",
            "calibration_percentile": self.calibration_percentile,
            "shrinkage": self.shrinkage,
            "threshold": self.threshold,
            "n_samples": int(len(X)),
            "n_classes": int(len(np.unique(y))),
            "calibration_scores": int(len(self.calibration_scores_)),
        }
        return self

    def score(self, X):
        return self._score_with(
            X, self.scaler, self.class_means, self.class_inv_covs
        )

    def score_one(self, x):
        return float(self.score(np.asarray(x).reshape(1, -1))[0])

    def is_ood(self, X):
        return self.score(X) > self.threshold

    def is_ood_one(self, x):
        return self.score_one(x) > self.threshold

    def summary(self):
        return dict(self.metadata)
