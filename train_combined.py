"""Train on combined Donate-a-Cry + InfantCry-DBL for balanced classes.

Donate-a-Cry:  457 clips (84% hungry — heavily imbalanced)
InfantCry-DBL: 337 clips (balanced ~20% per class)
Combined:      794 clips (more balanced)

Uses segment augmentation + SVM + SVMSMOTE.
"""
import glob
import os
import time
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import librosa
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, balanced_accuracy_score,
)
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SVMSMOTE
import joblib

DONATE_DIR = os.path.expanduser(
    "~/.cache/huggingface/hub/"
    "datasets--Nooon--Donate_a_cry/snapshots/"
    "491487c4258287aa2be7cc179b7928768be7008d"
)
DBL_DIR = "/Users/kl/Documents/Codex/2026-11-23/you-2/work/cry-agent/data/infantcry-dbl-v2/dunstan  337"
MODEL_DIR = os.path.expanduser("~/.config/cry-agent/models")

DONATE_MAP = {
    "belly_pain": "pain", "burping": "burp", "discomfort": "discomfort",
    "hungry": "hungry", "tired": "sleepy",
}
DUNSTAN_MAP = {
    "neh": "hungry", "owh": "sleepy", "heh": "discomfort",
    "eairh": "pain", "eh": "burp",
}
SR = 16000
SEG_DUR = 2.0


def extract_features(y, sr=SR):
    if sr != SR:
        y = librosa.resample(y, orig_sr=sr, target_sr=SR)
        sr = SR
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


def segments(y, sr=SR):
    seg_len = int(sr * SEG_DUR)
    if len(y) < seg_len:
        y = np.pad(y, (0, seg_len - len(y)))
        return [y]
    segs = []
    hop = seg_len // 2
    for s in range(0, len(y) - seg_len + 1, hop):
        segs.append(y[s:s+seg_len])
    return segs


def load_dataset():
    samples, labels = [], []
    sources = {"donate": 0, "dbl": 0}

    # Donate-a-Cry
    for cls_name in os.listdir(DONATE_DIR):
        cls_dir = os.path.join(DONATE_DIR, cls_name)
        if not os.path.isdir(cls_dir):
            continue
        mapped = DONATE_MAP.get(cls_name, cls_name)
        for wav in glob.glob(os.path.join(cls_dir, "*.wav")):
            try:
                y, sr = librosa.load(wav, sr=SR)
                for seg in segments(y, sr):
                    f = extract_features(seg, sr)
                    if f is not None:
                        samples.append(f)
                        labels.append(mapped)
                        sources["donate"] += 1
            except:
                pass

    # InfantCry-DBL
    for dunstan_label, mapped in DUNSTAN_MAP.items():
        cls_dir = os.path.join(DBL_DIR, dunstan_label)
        for wav in glob.glob(os.path.join(cls_dir, "*.wav")):
            try:
                y, sr = librosa.load(wav, sr=SR)
                for seg in segments(y, sr):
                    f = extract_features(seg, sr)
                    if f is not None:
                        samples.append(f)
                        labels.append(mapped)
                        sources["dbl"] += 1
            except:
                pass

    return np.array(samples), np.array(labels), sources


def train():
    print("=" * 60)
    print("COMBINED TRAINING (Donate-a-Cry + InfantCry-DBL)")
    print("=" * 60)

    print("\n[1/3] Loading combined dataset...")
    t0 = time.time()
    X, y, sources = load_dataset()
    print(f"  Segments: {len(X)}  Features: {X.shape[1]}  Time: {time.time()-t0:.0f}s")
    print(f"  Sources: {sources}")

    from collections import Counter
    print(f"  Distribution: {Counter(y)}")

    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    print(f"  Classes: {list(le.classes_)}")

    print(f"\n[2/3] Cross-validation...")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    y_pred = np.zeros(len(y_enc), dtype=int)

    for fold, (tr, te) in enumerate(skf.split(X, y_enc)):
        pipe = ImbPipeline([
            ("scaler", StandardScaler()),
            ("smote", SVMSMOTE(random_state=42,
                k_neighbors=min(3, min(np.bincount(y_enc[tr])) - 1))),
            ("clf", SVC(C=1.0, kernel="rbf", gamma="scale",
                class_weight="balanced", probability=True, random_state=42)),
        ])
        pipe.fit(X[tr], y_enc[tr])
        y_pred[te] = pipe.predict(X[te])

    acc = accuracy_score(y_enc, y_pred)
    bal = balanced_accuracy_score(y_enc, y_pred)
    print(f"  Accuracy:       {acc*100:.1f}%")
    print(f"  Balanced Acc:   {bal*100:.1f}%")
    print()
    print(classification_report(y_enc, y_pred, target_names=le.classes_, zero_division=0))
    cm = confusion_matrix(y_enc, y_pred)
    print("Confusion Matrix:")
    print(f"  {'':12s}", "  ".join(f"{c[:4]:>5s}" for c in le.classes_))
    for i, row in enumerate(cm):
        print(f"  {le.classes_[i]:12s}", "  ".join(f"{v:5d}" for v in row))

    print(f"\n[3/3] Training final + saving...")
    final = ImbPipeline([
        ("scaler", StandardScaler()),
        ("smote", SVMSMOTE(random_state=42,
            k_neighbors=min(3, min(np.bincount(y_enc)) - 1))),
        ("clf", SVC(C=1.0, kernel="rbf", gamma="scale",
            class_weight="balanced", probability=True, random_state=42)),
    ])
    final.fit(X, y_enc)

    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(final, os.path.join(MODEL_DIR, "cry_classifier.joblib"))
    joblib.dump(le, os.path.join(MODEL_DIR, "label_encoder.joblib"))

    import json
    config = {
        "model_type": "svm_rbf",
        "pipeline": "scaler + svm_smote + svc",
        "training_data": "donate-a-cry + infantcry-dbl",
        "n_features": X.shape[1],
        "classes": list(le.classes_),
        "cv_accuracy": round(acc, 3),
        "cv_balanced_accuracy": round(bal, 3),
        "sources": sources,
        "sample_rate": SR,
    }
    with open(os.path.join(MODEL_DIR, "model_config.json"), "w") as f:
        json.dump(config, f, indent=2)

    print(f"\n{'='*60}")
    print(f"DONE — Accuracy: {acc*100:.1f}%, Balanced: {bal*100:.1f}%")
    print(f"{'='*60}")


if __name__ == "__main__":
    train()
