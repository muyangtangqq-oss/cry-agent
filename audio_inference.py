"""Audio inference module — classify real cry audio.

Loads the trained SVM classifier and provides:
  - classify_file(path) -> {cause, confidence, probabilities}
  - classify_array(y, sr) -> same
  - get_cry_pattern_description(probabilities) -> text for the judge layer
"""
import os
import json
import warnings
warnings.filterwarnings("ignore")
from collections import deque

import numpy as np
import librosa
import joblib

from train_combined_v2 import extract_features

MODEL_DIR = os.path.expanduser("~/.config/cry-agent/models")
SR = 16000
SEGMENT_DURATION = 2.0


class CryAudioClassifier:
    def __init__(self, model_path=None, label_encoder_path=None,
                 segment_duration=None, sample_rate=None,
                 ood_path=None, ood_enabled=True, family_adapter=None):
        self.model_path = os.path.expanduser(
            model_path or os.path.join(MODEL_DIR, "cry_classifier.joblib")
        )
        self.label_encoder_path = os.path.expanduser(
            label_encoder_path or os.path.join(MODEL_DIR, "label_encoder.joblib")
        )
        self.sample_rate = int(sample_rate or SR)
        self.segment_duration = float(segment_duration or SEGMENT_DURATION)

        for path, hint in (
            (self.model_path, "audio classifier model"),
            (self.label_encoder_path, "label encoder"),
        ):
            if not os.path.exists(path):
                raise FileNotFoundError(
                    f"{hint} not found: {path}. Train one first "
                    "(see train_combined_v2.py) or point "
                    "signals.audio.model_path at an existing model."
                )

        self.pipeline = joblib.load(self.model_path)
        self.label_encoder = joblib.load(self.label_encoder_path)
        self.classes = list(self.label_encoder.classes_)
        config_path = os.path.join(
            os.path.dirname(self.model_path), "model_config.json"
        )
        self.model_config = {}
        if os.path.exists(config_path):
            with open(config_path) as f:
                self.model_config = json.load(f)
        self.feature_loudness = self.model_config.get(
            "feature_loudness", "peak"
        )
        self.file_cmvn = bool(self.model_config.get("file_cmvn", False))
        self.cmvn_min_segments = int(
            self.model_config.get("cmvn_min_segments", 2)
        )
        self.cmvn_window = int(self.model_config.get("cmvn_window", 8))
        self._cmvn_buffer = deque(maxlen=max(self.cmvn_window, 2))

        self.ood = None
        ood_path = os.path.expanduser(
            ood_path or os.path.join(
                os.path.dirname(self.model_path), "ood_detector.joblib"
            )
        )
        if ood_enabled and os.path.exists(ood_path):
            from ood import OODDetector
            self.ood = joblib.load(ood_path)
            self.ood_path = ood_path

        self.family_adapter = None
        if family_adapter is not None:
            self.set_family_adapter(family_adapter)

    def set_family_adapter(self, adapter):
        """Attach a per-family calibrator, rebinding it to this pipeline."""
        if hasattr(adapter, "bind"):
            adapter.bind(self.pipeline, self.pipeline.named_steps["scaler"])
        if [str(c) for c in adapter.classes] != [str(c) for c in self.classes]:
            raise ValueError(
                "family adapter classes do not match the audio model: "
                f"{list(adapter.classes)} vs {list(self.classes)}"
            )
        self.family_adapter = adapter
        return adapter

    def load_family_adapter(self, path):
        return self.set_family_adapter(joblib.load(os.path.expanduser(path)))

    def _extract_features(self, y, sr=None):
        return extract_features(
            y, sr or self.sample_rate, loudness=self.feature_loudness
        )

    def reset_session(self):
        self._cmvn_buffer.clear()

    def _normalize_features(self, features):
        """Apply file CMVN, or a rolling session buffer for live chunks."""
        if not self.file_cmvn or not features:
            return np.asarray(features), False
        arr = np.asarray(features, dtype=np.float32)
        if len(arr) >= self.cmvn_min_segments:
            return arr - arr.mean(axis=0, keepdims=True), True

        for feat in arr:
            self._cmvn_buffer.append(feat)
        if len(self._cmvn_buffer) >= self.cmvn_min_segments:
            mean = np.mean(np.stack(self._cmvn_buffer), axis=0)
            return arr - mean, True
        return arr, False

    def _segments(self, y, sr=None):
        seg_len = int((sr or self.sample_rate) * self.segment_duration)
        if len(y) < seg_len:
            y = np.pad(y, (0, seg_len - len(y)))
            return [y]
        segs = []
        hop = seg_len // 2
        for start in range(0, len(y) - seg_len + 1, hop):
            segs.append(y[start:start + seg_len])
        return segs

    def classify_array(self, y, sr):
        """Classify audio array. Returns aggregated result over segments."""
        segments = self._segments(y, sr)
        features_all = []
        for seg in segments:
            feat = self._extract_features(seg, sr)
            if feat is not None:
                features_all.append(feat)

        if not features_all:
            return {"cause": "unknown", "confidence": 0.0, "probabilities": {}}

        normalized, used_cmvn = self._normalize_features(features_all)
        features_all = list(normalized)

        if self.family_adapter is not None:
            probs_all = [
                self.family_adapter.predict_proba(np.array([feat]))[0]
                for feat in features_all
            ]
        else:
            probs_all = [
                self.pipeline.predict_proba(np.array([feat]))[0]
                for feat in features_all
            ]

        # Average probabilities across segments
        avg_probs = np.mean(probs_all, axis=0)
        best_idx = np.argmax(avg_probs)

        cause = self.classes[best_idx]
        confidence = float(avg_probs[best_idx])
        probabilities = {
            self.classes[i]: round(float(p), 3)
            for i, p in enumerate(avg_probs)
        }

        result = {
            "cause": cause,
            "confidence": round(confidence, 3),
            "probabilities": probabilities,
            "n_segments": len(probs_all),
            "cmvn": bool(used_cmvn),
            "cmvn_warmup": bool(self.file_cmvn and not used_cmvn),
        }
        if self.ood is not None and features_all:
            ood_scores = [self.ood.score_one(f) for f in features_all]
            worst = float(max(ood_scores))
            result["ood_score"] = round(worst, 3)
            result["ood_threshold"] = round(float(self.ood.threshold), 3)
            result["ood"] = bool(worst > self.ood.threshold)
        return result

    def classify_file(self, path):
        """Classify a .wav file."""
        self.reset_session()
        y, sr = librosa.load(path, sr=None)
        result = self.classify_array(y, sr)
        result["file"] = os.path.basename(path)
        return result

    def features_from_file(self, path):
        """Per-segment feature vectors, used by family calibration."""
        y, sr = librosa.load(path, sr=None)
        feats = []
        for seg in self._segments(y, sr):
            feat = self._extract_features(seg, sr)
            if feat is not None:
                feats.append(feat)
        if self.file_cmvn and len(feats) >= self.cmvn_min_segments:
            feats = list(
                np.asarray(feats) - np.asarray(feats).mean(
                    axis=0, keepdims=True
                )
            )
        return feats

    def get_cry_pattern(self, probabilities):
        """Convert probabilities to a text description for the judge."""
        sorted_p = sorted(probabilities.items(), key=lambda x: -x[1])
        top = sorted_p[:3]
        desc = ", ".join(f"{c}({p:.0%})" for c, p in top)
        return desc


if __name__ == "__main__":
    import glob

    DATA_DIR = os.path.expanduser(
        "~/.cache/huggingface/hub/"
        "datasets--Nooon--Donate_a_cry/snapshots/"
        "491487c4258287aa2be7cc179b7928768be7008d"
    )

    clf = CryAudioClassifier()
    print(f"Loaded model. Classes: {clf.classes}")
    print()

    # Test on one file per class
    test_files = []
    for class_dir in os.listdir(DATA_DIR):
        full = os.path.join(DATA_DIR, class_dir)
        if os.path.isdir(full):
            wavs = sorted(glob.glob(os.path.join(full, "*.wav")))
            if wavs:
                test_files.append((class_dir, wavs[0]))

    print(f"{'File':<50s} {'Predicted':>12s} {'Conf':>6s}  Top-3")
    print("-" * 100)

    correct = 0
    total = 0
    for true_label, wav_file in test_files:
        result = clf.classify_file(wav_file)
        mapped = {"belly_pain": "pain", "burping": "burp",
                  "discomfort": "discomfort", "hungry": "hungry",
                  "tired": "sleepy"}.get(true_label, true_label)
        is_correct = result["cause"] == mapped
        correct += is_correct
        total += 1

        top3 = clf.get_cry_pattern(result["probabilities"])
        mark = "✓" if is_correct else "✗"
        print(f" {mark} {result['file'][:48]:<48s} {result['cause']:>12s} "
              f"{result['confidence']:>5.0%}  {top3}")

    print(f"\nAccuracy on sample: {correct}/{total} ({correct/total*100:.0f}%)")
