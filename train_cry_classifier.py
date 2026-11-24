"""Train a cry classifier on Donate-a-Cry dataset.

Strategy for extreme imbalance (hungry: 382 vs burping: 8):
  - Segment augmentation: split each clip into 2s segments
  - Feature extraction per segment
  - SVM with class_weight='balanced'
  - Proper imblearn-aware cross-validation
"""
import glob
import os
import json
import time
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import librosa
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    balanced_accuracy_score,
)
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SVMSMOTE
import joblib

DATA_DIR = os.path.expanduser(
    "~/.cache/huggingface/hub/"
    "datasets--Nooon--Donate_a_cry/snapshots/"
    "491487c4258287aa2be7cc179b7928768be7008d"
)
MODEL_DIR = os.path.expanduser("~/.config/cry-agent/models")
LABEL_MAP = {
    "belly_pain": "pain",
    "burping": "burp",
    "discomfort": "discomfort",
    "hungry": "hungry",
    "tired": "sleepy",
}
SEGMENT_DURATION = 2.0  # seconds per segment
SR = 16000


def extract_features(y, sr=SR):
    """Extract MFCC + delta + spectral + pitch features."""
    if len(y) < sr * 0.3:
        return None
    y = y / (np.max(np.abs(y)) + 1e-8)

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    mfcc_mean = mfcc.mean(axis=1)
    mfcc_std = mfcc.std(axis=1)

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

    features = np.concatenate([
        mfcc_mean, mfcc_std,
        delta1.mean(axis=1), delta2.mean(axis=1),
        [centroid.mean(), centroid.std()],
        [bandwidth.mean(), bandwidth.std()],
        [rolloff.mean(), rolloff.std()],
        [flatness.mean(), flatness.std()],
        [zcr.mean(), zcr.std()],
        [rms.mean(), rms.std()],
        [f0_mean, f0_std, voiced_ratio],
    ])
    return features


def audio_to_segments(y, sr, seg_dur=SEGMENT_DURATION):
    """Split audio into overlapping segments for augmentation."""
    seg_len = int(sr * seg_dur)
    if len(y) < seg_len:
        # Pad if too short
        y = np.pad(y, (0, seg_len - len(y)))
        return [y]

    segments = []
    hop = seg_len // 2  # 50% overlap
    for start in range(0, len(y) - seg_len + 1, hop):
        segments.append(y[start:start + seg_len])
    return segments


def load_dataset():
    """Load audio files, split into segments, extract features."""
    samples = []
    labels = []
    file_counts = {}

    for class_name in os.listdir(DATA_DIR):
        if class_name in ("README.md", ".gitattributes"):
            continue
        class_dir = os.path.join(DATA_DIR, class_name)
        if not os.path.isdir(class_dir):
            continue

        mapped = LABEL_MAP.get(class_name, class_name)
        wav_files = glob.glob(os.path.join(class_dir, "*.wav"))
        file_counts[mapped] = len(wav_files)

        for wav_file in wav_files:
            try:
                y, sr = librosa.load(wav_file, sr=SR)
                segments = audio_to_segments(y, sr)
                for seg in segments:
                    feat = extract_features(seg, sr)
                    if feat is not None:
                        samples.append(feat)
                        labels.append(mapped)
            except Exception as e:
                print(f"    SKIP {os.path.basename(wav_file)}: {e}")

    print(f"  Original files: {file_counts}")
    return np.array(samples), np.array(labels)


def cross_val_eval(X, y_encoded, le):
    """Manual cross-validation with SMOTE inside folds."""
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    y_pred = np.zeros(len(y_encoded), dtype=int)

    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y_encoded)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y_encoded[train_idx], y_encoded[test_idx]

        # Pipeline: scale -> SVMSMOTE -> SVM
        pipe = ImbPipeline([
            ("scaler", StandardScaler()),
            ("smote", SVMSMOTE(
                random_state=42,
                k_neighbors=min(3, min(np.bincount(y_train)) - 1),
            )),
            ("clf", SVC(
                C=1.0,
                kernel="rbf",
                gamma="scale",
                class_weight="balanced",
                probability=True,
                random_state=42,
            )),
        ])

        pipe.fit(X_train, y_train)
        y_pred[test_idx] = pipe.predict(X_test)

    acc = accuracy_score(y_encoded, y_pred)
    bal_acc = balanced_accuracy_score(y_encoded, y_pred)

    print(f"  CV Accuracy:       {acc*100:.1f}%")
    print(f"  Balanced Accuracy: {bal_acc*100:.1f}%")
    print()
    print(classification_report(
        y_encoded, y_pred,
        target_names=le.classes_,
        zero_division=0,
    ))

    cm = confusion_matrix(y_encoded, y_pred)
    print("Confusion Matrix:")
    print(f"  {'':15s}", "  ".join(f"{c[:4]:>5s}" for c in le.classes_))
    for i, row in enumerate(cm):
        print(f"  {le.classes_[i]:15s}", "  ".join(f"{v:5d}" for v in row))

    return acc, bal_acc


def train():
    print("=" * 60)
    print("CRY CLASSIFIER (SVM + SVMSMOTE + segment augmentation)")
    print("=" * 60)

    print("\n[1/3] Loading dataset with segment augmentation...")
    t0 = time.time()
    X, y = load_dataset()
    print(f"  Segments: {len(X)}, Features: {X.shape[1]}, Time: {time.time()-t0:.1f}s")

    from collections import Counter
    print(f"  Segment distribution: {Counter(y)}")

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    print(f"  Classes: {list(le.classes_)}")

    print(f"\n[2/3] Cross-validation...")
    acc, bal_acc = cross_val_eval(X, y_encoded, le)

    print(f"\n[3/3] Training final model on all data...")
    final_pipe = ImbPipeline([
        ("scaler", StandardScaler()),
        ("smote", SVMSMOTE(
            random_state=42,
            k_neighbors=min(3, min(np.bincount(y_encoded)) - 1),
        )),
        ("clf", SVC(
            C=1.0,
            kernel="rbf",
            gamma="scale",
            class_weight="balanced",
            probability=True,
            random_state=42,
        )),
    ])
    final_pipe.fit(X, y_encoded)

    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(final_pipe, os.path.join(MODEL_DIR, "cry_classifier.joblib"))
    joblib.dump(le, os.path.join(MODEL_DIR, "label_encoder.joblib"))

    config = {
        "model_type": "svm_rbf",
        "pipeline": "standard_scaler + svm_smote + svc",
        "features": "mfcc(20)+delta1+delta2+spectral+pitch",
        "n_features": X.shape[1],
        "classes": list(le.classes_),
        "cv_accuracy": round(acc, 3),
        "cv_balanced_accuracy": round(bal_acc, 3),
        "label_map": LABEL_MAP,
        "sample_rate": SR,
        "segment_duration": SEGMENT_DURATION,
    }
    with open(os.path.join(MODEL_DIR, "model_config.json"), "w") as f:
        json.dump(config, f, indent=2)

    print(f"\n{'='*60}")
    print(f"DONE — Accuracy: {acc*100:.1f}%, Balanced: {bal_acc*100:.1f}%")
    print(f"Model saved to {MODEL_DIR}")
    print(f"{'='*60}")


if __name__ == "__main__":
    train()
