"""Audio inference module — classify real cry audio.

Loads the trained SVM classifier and provides:
  - classify_file(path) -> {cause, confidence, probabilities}
  - classify_array(y, sr) -> same
  - get_cry_pattern_description(probabilities) -> text for the judge layer
"""
import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import librosa
import joblib

MODEL_DIR = os.path.expanduser("~/.config/cry-agent/models")
SR = 16000
SEGMENT_DURATION = 2.0


class CryAudioClassifier:
    def __init__(self, model_path=None, label_encoder_path=None,
                 segment_duration=None, sample_rate=None):
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

    def _extract_features(self, y, sr=None):
        sr = sr or self.sample_rate
        if sr != self.sample_rate:
            y = librosa.resample(y, orig_sr=sr, target_sr=self.sample_rate)
            sr = self.sample_rate
        if len(y) < sr * 0.3:
            return None
        y = y / (np.max(np.abs(y)) + 1e-8)

        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
        delta1 = librosa.feature.delta(mfcc)
        delta2 = librosa.feature.delta(mfcc, order=2)
        centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
        bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
        rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
        flatness = librosa.feature.spectral_flatness(y=y)
        zcr = librosa.feature.zero_crossing_rate(y)
        rms = librosa.feature.rms(y=y)
        f0, voiced, _ = librosa.pyin(y=y, sr=sr, fmin=80, fmax=500)
        f0_v = f0[voiced] if voiced.any() else np.array([0])
        f0_mean = f0_v.mean() if len(f0_v) > 0 else 0
        f0_std = f0_v.std() if len(f0_v) > 0 else 0
        voiced_ratio = voiced.sum() / len(voiced) if len(voiced) > 0 else 0

        return np.concatenate([
            mfcc.mean(axis=1), mfcc.std(axis=1),
            delta1.mean(axis=1), delta2.mean(axis=1),
            [centroid.mean(), centroid.std()],
            [bandwidth.mean(), bandwidth.std()],
            [rolloff.mean(), rolloff.std()],
            [flatness.mean(), flatness.std()],
            [zcr.mean(), zcr.std()],
            [rms.mean(), rms.std()],
            [f0_mean, f0_std, voiced_ratio],
        ])

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
        probs_all = []
        for seg in segments:
            feat = self._extract_features(seg, sr)
            if feat is not None:
                proba = self.pipeline.predict_proba([feat])[0]
                probs_all.append(proba)

        if not probs_all:
            return {"cause": "unknown", "confidence": 0.0, "probabilities": {}}

        # Average probabilities across segments
        avg_probs = np.mean(probs_all, axis=0)
        best_idx = np.argmax(avg_probs)

        cause = self.classes[best_idx]
        confidence = float(avg_probs[best_idx])
        probabilities = {
            self.classes[i]: round(float(p), 3)
            for i, p in enumerate(avg_probs)
        }

        return {
            "cause": cause,
            "confidence": round(confidence, 3),
            "probabilities": probabilities,
            "n_segments": len(probs_all),
        }

    def classify_file(self, path):
        """Classify a .wav file."""
        y, sr = librosa.load(path, sr=None)
        result = self.classify_array(y, sr)
        result["file"] = os.path.basename(path)
        return result

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
