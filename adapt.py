"""Lightweight domain adaptation for family-level calibration.

The global classifier is trained on public corpora and collapses when the
recording setup changes (see reports/loco_cv.md). Full retraining is too
expensive for a 3-5 family pilot, so this module implements three small
adapters that can run on a device:

  * ``PriorCorrector``   — undo the source corpus class prior
  * ``PrototypeAdapter`` — k labelled clips per class -> nearest centroid
  * ``CoralAdapter``     — unlabeled family audio -> feature-space alignment

``blend_proba`` combines a global model with a prototype model.
"""
from collections import defaultdict

import numpy as np
from sklearn.metrics import balanced_accuracy_score


def _normalize_rows(proba):
    proba = np.clip(np.asarray(proba, dtype=float), 0.0, None)
    total = proba.sum(axis=1, keepdims=True)
    total[total <= 0] = 1.0
    return proba / total


def blend_proba(global_proba, adapter_proba, alpha=0.5):
    """Weighted average of two aligned probability matrices."""
    return _normalize_rows(
        alpha * _normalize_rows(global_proba)
        + (1.0 - alpha) * _normalize_rows(adapter_proba)
    )


class PriorCorrector:
    """Correct a classifier trained on a skewed corpus to a target prior."""

    def __init__(self, classes):
        self.classes = np.asarray(classes)

    def fit(self, y_source, target_prior=None):
        counts = np.array(
            [(np.asarray(y_source) == c).sum() for c in self.classes],
            dtype=float,
        )
        self.source_prior = np.clip(counts / counts.sum(), 1e-6, None)
        if target_prior is None:
            self.target_prior = np.full(len(self.classes), 1.0 / len(self.classes))
        else:
            target_prior = np.asarray(target_prior, dtype=float)
            self.target_prior = target_prior / target_prior.sum()
        return self

    def transform(self, proba):
        ratio = self.target_prior / self.source_prior
        return _normalize_rows(np.asarray(proba) * ratio)


class PrototypeAdapter:
    """Nearest-centroid classifier over k labelled clips per class.

    Cosine distance is used because the feature vector mixes MFCC means,
    spectral shape, and pitch statistics with very different scales.
    """

    def __init__(self, classes, metric="cosine", temperature=0.1,
                 shrink=0.15):
        self.classes = np.asarray(classes)
        self.metric = metric
        self.temperature = temperature
        self.shrink = shrink

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        self.centroids = []
        self.global_centroid = X.mean(axis=0)
        for cls in self.classes:
            rows = X[y == cls]
            if len(rows) == 0:
                self.centroids.append(self.global_centroid)
                continue
            centroid = rows.mean(axis=0)
            if self.shrink:
                centroid = (
                    (1.0 - self.shrink) * centroid
                    + self.shrink * self.global_centroid
                )
            self.centroids.append(centroid)
        self.centroids = np.stack(self.centroids)
        return self

    def _distances(self, X):
        X = np.asarray(X, dtype=float)
        if self.metric == "cosine":
            Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)
            Cn = self.centroids / (
                np.linalg.norm(self.centroids, axis=1, keepdims=True) + 1e-9
            )
            return 1.0 - Xn @ Cn.T
        diff = X[:, None, :] - self.centroids[None, :, :]
        return np.sqrt((diff ** 2).sum(axis=2))

    def predict_proba(self, X):
        dist = self._distances(X)
        logits = -dist / max(self.temperature, 1e-6)
        logits -= logits.max(axis=1, keepdims=True)
        proba = np.exp(logits)
        return _normalize_rows(proba)

    def predict(self, X):
        return self.classes[np.argmax(self.predict_proba(X), axis=1)]


def _sqrtm_psd(matrix, inverse=False, eps=1e-3):
    values, vectors = np.linalg.eigh(matrix)
    values = np.clip(values, eps, None)
    if inverse:
        values = 1.0 / np.sqrt(values)
    else:
        values = np.sqrt(values)
    return (vectors * values) @ vectors.T


class CoralAdapter:
    """Align target feature mean/covariance to the source distribution.

    Unsupervised: only needs unlabeled audio from the family. With few
    target samples the covariance is regularized toward the identity.
    """

    def __init__(self, eps=1e-2):
        self.eps = eps

    def fit(self, X_source, X_target):
        X_source = np.asarray(X_source, dtype=float)
        return self._fit_from_moments(
            X_source.mean(axis=0), np.cov(X_source, rowvar=False), X_target
        )

    def fit_from_stats(self, mean_source, cov_source, X_target):
        """Fit when only compact source statistics were shipped."""
        return self._fit_from_moments(
            mean_source, cov_source, X_target
        )

    def _fit_from_moments(self, mean_source, cov_source, X_target):
        X_target = np.asarray(X_target, dtype=float)
        mean_source = np.asarray(mean_source, dtype=float)
        cov_source = np.asarray(cov_source, dtype=float)
        self.mu_source = mean_source
        self.mu_target = X_target.mean(axis=0)
        dim = X_target.shape[1]
        cs = cov_source + self.eps * np.eye(dim)
        ct = np.cov(X_target, rowvar=False) + self.eps * np.eye(dim)
        self.transform_matrix = _sqrtm_psd(ct, inverse=True) @ _sqrtm_psd(cs)
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        return (X - self.mu_target) @ self.transform_matrix + self.mu_source


class FamilyCalibrator:
    """Frozen adapter built from a family's own labelled clips.

    The selection mirrors ``test_adaptation.py``: a recording-level
    validation slice chooses between the global model, unsupervised
    CORAL alignment, nearest-centroid prototypes, and CORAL + prototypes.
    """

    def __init__(self, model, scaler, classes, method="global", alpha=0.5,
                 prototype_metric="cosine"):
        self.model = model
        self.scaler = scaler
        self.classes = [str(c) for c in classes]
        self.method = method
        self.alpha = float(alpha)
        self.prototype_metric = prototype_metric
        self.coral = None
        self.prototypes = None
        self.shots = 0
        self.validation = {}

    def bind(self, model, scaler):
        """Re-point at the live classifier after loading from disk."""
        self.model = model
        self.scaler = scaler
        return self

    def __getstate__(self):
        # The SVM and scaler already live with the audio model; storing a
        # second copy would turn a 50 KB adapter into an 18 MB one.
        state = self.__dict__.copy()
        state["model"] = None
        state["scaler"] = None
        return state

    def _scaled(self, X):
        return self.scaler.transform(np.atleast_2d(np.asarray(X, dtype=float)))

    def predict_proba(self, X):
        if self.model is None or self.scaler is None:
            raise RuntimeError(
                "FamilyCalibrator is not bound; call bind(model, scaler)"
            )
        Xs = self._scaled(X)
        base = self.model.predict_proba(Xs)
        if self.prototypes is None:
            if self.method == "coral" and self.coral is not None:
                return self.model.predict_proba(self.coral.transform(Xs))
            return base
        if self.method == "proto":
            return self.prototypes.predict_proba(Xs)
        if self.method == "coral-proto" and self.coral is not None:
            return self.prototypes.predict_proba(self.coral.transform(Xs))
        if self.method == "blend":
            return blend_proba(
                base, self.prototypes.predict_proba(Xs), self.alpha
            )
        return base

    @classmethod
    def fit(cls, model, scaler, classes, X_labeled, y_labeled, files=None,
            X_unlabeled=None, source_mean=None, source_cov=None,
            max_shots=20, cal_fraction=0.75, seed=0,
            prototype_metric="cosine"):
        classes = [str(c) for c in classes]
        index = {c: i for i, c in enumerate(classes)}
        X_labeled = np.asarray(X_labeled, dtype=float)

        def encode(value):
            name = str(value)
            if name in index:
                return index[name]
            try:
                pos = int(value)
                if 0 <= pos < len(classes):
                    return pos
            except (TypeError, ValueError):
                pass
            return -1

        y = np.array([encode(v) for v in y_labeled], dtype=int)
        keep = y >= 0
        X_labeled, y = X_labeled[keep], y[keep]
        if files is not None:
            files = np.asarray([str(f) for f in files])[keep]

        calibrator = cls(
            model, scaler, classes, prototype_metric=prototype_metric
        )
        if len(X_labeled) == 0:
            return calibrator

        Xs = scaler.transform(X_labeled)
        if (X_unlabeled is not None and np.asarray(X_unlabeled).size
                and source_mean is not None and source_cov is not None):
            target = scaler.transform(
                np.atleast_2d(np.asarray(X_unlabeled, dtype=float))
            )
            calibrator.coral = CoralAdapter().fit_from_stats(
                source_mean, source_cov, target
            )

        counts = np.bincount(y, minlength=len(classes))
        present = counts[counts > 0]
        shots = int(min(max_shots, present.min())) if len(present) else 0
        calibrator.shots = shots

        fit_idx, val_idx = cls._split(files, y, cal_fraction, seed)
        if len(val_idx) == 0:
            calibrator.method = "coral" if calibrator.coral else "global"
            return calibrator

        candidates = {
            "global": model.predict_proba(Xs[val_idx]),
        }
        if calibrator.coral is not None:
            candidates["coral"] = model.predict_proba(
                calibrator.coral.transform(Xs[val_idx])
            )
        if shots and len(np.unique(y[fit_idx])) > 1:
            shot_fit = cls._sample_shots(
                y, files, fit_idx, shots, seed
            )
            proto = PrototypeAdapter(
                list(range(len(classes))), metric=prototype_metric
            ).fit(Xs[shot_fit], y[shot_fit])
            candidates["proto"] = proto.predict_proba(Xs[val_idx])
            if calibrator.coral is not None:
                coral_proto = PrototypeAdapter(
                    list(range(len(classes))), metric=prototype_metric
                ).fit(
                    calibrator.coral.transform(Xs[shot_fit]), y[shot_fit]
                )
                candidates["coral-proto"] = coral_proto.predict_proba(
                    calibrator.coral.transform(Xs[val_idx])
                )

        calibrator.validation = {
            name: round(float(balanced_accuracy_score(
                y[val_idx], np.argmax(proba, axis=1)
            )), 3)
            for name, proba in candidates.items()
        }
        best = max(
            calibrator.validation,
            key=lambda name: (
                calibrator.validation[name],
                candidates[name].max(axis=1).mean(),
            ),
        )
        calibrator.method = best
        if best in ("proto", "coral-proto", "blend") and shots:
            shot_all = cls._sample_shots(y, files, np.arange(len(y)), shots, seed)
            space = Xs
            if best == "coral-proto" and calibrator.coral is not None:
                space = calibrator.coral.transform(Xs)
            calibrator.prototypes = PrototypeAdapter(
                list(range(len(classes))), metric=prototype_metric
            ).fit(space[shot_all], y[shot_all])
        return calibrator

    @staticmethod
    def _sample_shots(y, files, idx, shots, seed):
        """Up to `shots` clips per class, preferring distinct recordings."""
        rng = np.random.default_rng(seed)
        picked = []
        for cls in np.unique(y[idx]):
            rows = idx[y[idx] == cls]
            if files is None:
                take = min(shots, len(rows))
                picked.extend(
                    rng.choice(rows, size=take, replace=False).tolist()
                )
                continue
            by_file = defaultdict(list)
            for i in rows:
                by_file[str(files[i])].append(int(i))
            names = sorted(by_file)
            rng.shuffle(names)
            chosen = []
            for name in names:
                if len(chosen) >= shots:
                    break
                chosen.append(int(rng.choice(by_file[name])))
            if len(chosen) < shots:
                remaining = [int(i) for i in rows if int(i) not in chosen]
                extra = min(shots - len(chosen), len(remaining))
                if extra:
                    chosen.extend(
                        rng.choice(remaining, size=extra, replace=False).tolist()
                    )
            picked.extend(chosen)
        return np.array(picked, dtype=int)

    @staticmethod
    def _split(files, y, cal_fraction, seed):
        if files is None or len(files) == 0:
            return np.array([], dtype=int), np.array([], dtype=int)
        rng = np.random.default_rng(seed)
        fit_idx, val_idx = [], []
        for cls in np.unique(y):
            rows = np.where(y == cls)[0]
            names = sorted({str(files[i]) for i in rows})
            if len(names) < 2:
                return np.array([], dtype=int), np.array([], dtype=int)
            rng.shuffle(names)
            n_fit = max(1, int(round(len(names) * cal_fraction)))
            n_fit = min(n_fit, len(names) - 1)
            fit_files = set(names[:n_fit])
            fit_idx.extend(
                int(i) for i in rows if str(files[i]) in fit_files
            )
            val_idx.extend(
                int(i) for i in rows if str(files[i]) not in fit_files
            )
        return np.array(fit_idx, dtype=int), np.array(val_idx, dtype=int)
