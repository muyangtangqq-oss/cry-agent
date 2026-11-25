"""Tests for channel augmentation and recording-level feature normalization."""
from collections import deque

import numpy as np

from audio_inference import CryAudioClassifier
from augment import channel_augment
from feature_norm import (
    apply_file_means,
    apply_owner_means,
    file_cmvn,
    file_mean_map,
)
from train_combined_v2 import SR, extract_features


def test_file_cmvn_removes_recording_mean():
    X = np.array([[1.0, 2.0], [3.0, 4.0], [10.0, 20.0]], dtype=np.float32)
    files = ["a.wav", "a.wav", "b.wav"]
    out, covered = file_cmvn(X, files)
    assert covered == 2
    assert np.allclose(out[:2].mean(axis=0), 0.0)
    assert np.allclose(out[2], X[2])


def test_owner_means_match_file_means():
    X = np.array([[1.0], [3.0], [10.0]], dtype=np.float32)
    files = ["a.wav", "a.wav", "b.wav"]
    means = file_mean_map(X, files)
    assert np.allclose(means["a.wav"], [2.0])
    assert means["b.wav"] is None

    owners = [0, 0, 2]
    variants = np.array([[5.0], [7.0], [11.0]], dtype=np.float32)
    out = apply_owner_means(variants, owners, files, means)
    assert np.allclose(out[:, 0], [3.0, 5.0, 11.0])


def test_channel_augment_is_finite_and_changes_signal():
    rng = np.random.default_rng(0)
    y = rng.normal(0.0, 0.05, SR * 2).astype(np.float32)
    out = channel_augment(y, SR, seed=7)
    assert out.shape == y.shape
    assert np.isfinite(out).all()
    assert not np.allclose(out, y)


def test_rms_loudness_features_are_finite():
    rng = np.random.default_rng(0)
    y = rng.normal(0.0, 0.05, SR * 2).astype(np.float32)
    feat = extract_features(y, SR, loudness="rms")
    assert feat is not None
    assert np.isfinite(feat).all()


def test_classify_array_normalizes_before_predicting():
    captured = []
    clf = CryAudioClassifier.__new__(CryAudioClassifier)
    clf.file_cmvn = True
    clf.cmvn_min_segments = 2
    clf._cmvn_buffer = deque(maxlen=8)
    clf.family_adapter = None
    clf.ood = None
    clf.classes = ["hungry", "pain"]
    clf._segments = lambda y, sr: [np.zeros(2), np.zeros(2)]
    feats = iter([np.array([1.0, 2.0]), np.array([3.0, 4.0])])
    clf._extract_features = lambda seg, sr: next(feats)

    class FakePipeline:
        def predict_proba(self, X):
            captured.append(np.asarray(X)[0].copy())
            return np.array([[0.7, 0.3]])

    clf.pipeline = FakePipeline()
    result = clf.classify_array(np.zeros(4, dtype=np.float32), SR)

    assert result["cmvn"] is True
    assert len(captured) == 2
    assert np.allclose(captured[0], [-1.0, -1.0])
    assert np.allclose(captured[1], [1.0, 1.0])


def test_ood_scores_raw_features_when_configured():
    seen = []
    clf = CryAudioClassifier.__new__(CryAudioClassifier)
    clf.file_cmvn = True
    clf.ood_on_raw = True
    clf.cmvn_min_segments = 2
    clf._cmvn_buffer = deque(maxlen=8)
    clf.family_adapter = None
    clf.classes = ["hungry", "pain"]
    clf._segments = lambda y, sr: [np.zeros(2), np.zeros(2)]
    feats = iter([np.array([1.0, 2.0]), np.array([3.0, 4.0])])
    clf._extract_features = lambda seg, sr: next(feats)

    class FakePipeline:
        def predict_proba(self, X):
            return np.array([[0.7, 0.3]])

    class FakeOOD:
        threshold = 0.5

        def score_one(self, feat):
            seen.append(np.asarray(feat).copy())
            return 0.1

    clf.pipeline = FakePipeline()
    clf.ood = FakeOOD()
    result = clf.classify_array(np.zeros(4, dtype=np.float32), SR)

    assert result["cmvn"] is True
    assert result["ood"] is False
    assert np.allclose(seen[0], [1.0, 2.0])
    assert np.allclose(seen[1], [3.0, 4.0])


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"  ok {name}")
    print("Feature normalization tests passed.")
