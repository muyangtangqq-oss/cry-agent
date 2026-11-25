"""Train the cry classifier with minority-class audio augmentation.

Compares the same stratified CV split with and without augmentation, then
saves the augmented model to a separate directory so the production model
is only replaced when the numbers justify it.

Usage:
  python train_augmented.py
  python train_augmented.py --target-per-class 300 --promote
"""
import argparse
import glob
import json
import os
import time
import warnings
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor

warnings.filterwarnings("ignore")

import joblib
import librosa
import numpy as np
from imblearn.over_sampling import SMOTE, SVMSMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC

from augment import augment, channel_augment
from feature_norm import apply_file_means, apply_owner_means, file_mean_map
from train_combined_v2 import (
    DBL_DIR,
    DONATE_DIR,
    DONATE_MAP,
    DS_DIR,
    DUNSTAN_MAP,
    SR,
    extract_features,
    segs,
)

DEFAULT_MODEL_DIR = os.path.expanduser("~/.config/cry-agent/models")
OUTPUT_DIR = os.path.expanduser("~/.config/cry-agent/models_aug")


def _iter_class_dirs(base, mapping, source_name):
    if not os.path.isdir(base):
        print(f"  skip missing extra data dir: {base}")
        return
    for class_name in sorted(os.listdir(base)):
        class_dir = os.path.join(base, class_name)
        if not os.path.isdir(class_dir):
            continue
        label = mapping.get(class_name, class_name) if mapping else class_name
        for wav in sorted(glob.glob(os.path.join(class_dir, "*.wav"))):
            try:
                y, sr = librosa.load(wav, sr=SR)
            except Exception:
                continue
            for seg in segs(y, sr):
                yield seg.astype(np.float32), label, source_name, wav


def iter_raw_segments(extra_dirs=None):
    sources = [
        ("donate", DONATE_DIR, DONATE_MAP),
        ("dbl", DBL_DIR, DUNSTAN_MAP),
        ("dynamic", DS_DIR, None),
    ]
    for source_name, base, mapping in sources:
        yield from _iter_class_dirs(base, mapping, source_name)
    for idx, extra in enumerate(extra_dirs or []):
        yield from _iter_class_dirs(extra, None, f"extra{idx}")


def load_raw(extra_dirs=None):
    raw, labels, sources, files = [], [], [], []
    for seg, label, source, path in iter_raw_segments(extra_dirs):
        raw.append(seg)
        labels.append(label)
        sources.append(source)
        files.append(path)
    return raw, np.array(labels), sources, files


def _featurize_one(args):
    y, sr, loudness = args
    return extract_features(y, sr, loudness=loudness)


def featurize(raw, jobs, loudness="peak"):
    payload = [(y, SR, loudness) for y in raw]
    if jobs <= 1:
        return [_featurize_one(p) for p in payload]
    with ProcessPoolExecutor(max_workers=jobs) as pool:
        return list(pool.map(_featurize_one, payload, chunksize=8))


def make_pipeline(labels, sampler="svmsmote", probability=True):
    counts = np.bincount(labels)
    counts = counts[counts > 0]
    k = max(1, min(3, int(counts.min()) - 1))
    steps = [("scaler", StandardScaler())]
    if sampler == "smote":
        steps.append(("smote", SMOTE(random_state=42, k_neighbors=k)))
    elif sampler == "svmsmote":
        steps.append(("smote", SVMSMOTE(random_state=42, k_neighbors=k)))
    steps.append(
        ("clf", SVC(
            C=1.0, kernel="rbf", gamma="scale",
            class_weight="balanced", probability=probability, random_state=42,
        ))
    )
    return ImbPipeline(steps)


def build_variants(raw, labels, target_per_class, max_per_source, jobs,
                   seed=42, loudness="peak", files=None, file_means=None):
    """Generate augmented features keyed by source index (training-only use)."""
    indices = defaultdict(list)
    for idx, label in enumerate(labels):
        indices[label].append(idx)

    variant_raw = defaultdict(list)
    generated = Counter()
    for label, idxs in indices.items():
        need = max(0, target_per_class - len(idxs))
        if not need:
            continue
        per_source = min(max_per_source, int(np.ceil(need / len(idxs))))
        for source_idx in idxs:
            for k in range(per_source):
                variant_raw[source_idx].append(
                    augment(raw[source_idx], SR, seed=seed + source_idx * 1009 + k)
                )
                generated[label] += 1

    flat, owners = [], []
    for source_idx, variants in variant_raw.items():
        for variant in variants:
            flat.append(variant)
            owners.append(source_idx)

    print(f"  augmented copies: {len(flat)}  "
          f"by class: {dict(sorted(generated.items()))}")
    feats = featurize(flat, jobs, loudness) if flat else []
    valid = [(o, f) for o, f in zip(owners, feats) if f is not None]
    owners = [o for o, _ in valid]
    feats = [f for _, f in valid]
    if feats and file_means is not None and files is not None:
        feats = apply_owner_means(feats, owners, files, file_means)

    by_source = defaultdict(list)
    for owner, feat in zip(owners, feats):
        if feat is not None:
            by_source[owner].append(feat)
    return by_source, generated


def build_channel_variants(raw, jobs, copies=1, loudness="peak",
                           files=None, file_means=None):
    """One or more channel-augmented copies of every training segment."""
    if copies <= 0:
        return {}, 0
    flat, owners = [], []
    for idx, segment in enumerate(raw):
        for copy in range(copies):
            flat.append(channel_augment(
                segment, SR, seed=idx * 104729 + 17 + copy * 7919
            ))
            owners.append(idx)
    feats = featurize(flat, jobs, loudness)
    valid = [(o, f) for o, f in zip(owners, feats) if f is not None]
    owners = [o for o, _ in valid]
    feats = [f for _, f in valid]
    if file_means is not None and files is not None:
        feats = apply_owner_means(feats, owners, files, file_means)
    by_source = defaultdict(list)
    for owner, feat in zip(owners, feats):
        if feat is not None:
            by_source[owner].append(feat)
    return by_source, len(flat)


def expand_train(train_idx, base_feats, labels, variants):
    X = [base_feats[i] for i in train_idx]
    y = [labels[i] for i in train_idx]
    for i in train_idx:
        for feat in variants.get(i, []):
            X.append(feat)
            y.append(labels[i])
    return np.asarray(X), np.asarray(y)


def evaluate(y_true, pred):
    return {
        "accuracy": accuracy_score(y_true, pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, pred),
    }


def promote_model(src_dir, dst_dir=DEFAULT_MODEL_DIR):
    import shutil
    os.makedirs(dst_dir, exist_ok=True)
    optional = {"source_stats.npz"}
    for name in ("cry_classifier.joblib", "label_encoder.joblib",
                 "model_config.json", "source_stats.npz"):
        src = os.path.join(src_dir, name)
        if not os.path.exists(src):
            if name in optional:
                continue
            raise FileNotFoundError(src)
        dst = os.path.join(dst_dir, name)
        if os.path.exists(dst):
            shutil.copy2(dst, dst + ".pre_aug.bak")
        shutil.copy2(src, dst)
    print(f"  promoted {src_dir} -> {dst_dir} "
          "(previous files kept as *.pre_aug.bak)")


def run(args):
    t0 = time.time()
    print("=" * 72)
    print("AUGMENTED TRAINING — minority-class pitch/stretch/gain/noise")
    print("=" * 72)
    print("[1/5] Loading raw segments...")
    raw, labels, sources, files = load_raw(args.extra_data)
    counts = Counter(labels)
    print(f"  segments={len(raw)} sources={len(set(files))} "
          f"time={time.time() - t0:.0f}s")
    print(f"  distribution: {dict(sorted(counts.items()))}")

    jobs = args.jobs or max(1, (os.cpu_count() or 2) - 1)
    print(f"\n[2/5] Extracting base features ({jobs} workers)...")
    t1 = time.time()
    loudness = "rms" if args.rms_loudness else "peak"
    base_feats = featurize(raw, jobs, loudness)
    file_means = None
    if args.file_cmvn:
        file_means = file_mean_map(base_feats, files)
        base_feats = apply_file_means(base_feats, files, file_means)
        covered = sum(
            1 for path in files if file_means.get(path) is not None
        )
        print(f"  CMVN: {covered}/{len(files)} segments normalized")
    print(f"  done in {time.time() - t1:.0f}s")

    print("\n[3/5] Generating augmented variants...")
    t2 = time.time()
    variants, generated = build_variants(
        raw, labels, args.target_per_class, args.max_per_source, jobs,
        loudness=loudness, files=files, file_means=file_means,
    )
    channel_copies = 0
    if args.channel_augment:
        channel_variants, channel_copies = build_channel_variants(
            raw, jobs, args.channel_copies, loudness, files, file_means
        )
        for idx, feats in channel_variants.items():
            variants[idx].extend(feats)
        print(f"  channel-augmented copies: {channel_copies}")
    print(f"  done in {time.time() - t2:.0f}s")

    le = LabelEncoder()
    ye = le.fit_transform(labels)
    min_count = min(Counter(ye).values())
    n_splits = max(2, min(args.folds, min_count))
    print(f"\n[4/5] {n_splits}-fold CV (smallest class has {min_count} segments)")

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    base_pred = np.zeros(len(ye), dtype=int)
    aug_pred = np.zeros(len(ye), dtype=int)

    for fold, (train_idx, test_idx) in enumerate(skf.split(np.zeros(len(ye)), ye), 1):
        X_base = np.asarray([base_feats[i] for i in train_idx])
        X_aug, y_aug = expand_train(train_idx, base_feats, ye, variants)

        base_pipe = make_pipeline(
            ye[train_idx], args.sampler, probability=False
        )
        base_pipe.fit(X_base, ye[train_idx])
        base_pred[test_idx] = base_pipe.predict(
            np.asarray([base_feats[i] for i in test_idx])
        )

        aug_pipe = make_pipeline(y_aug, args.sampler, probability=False)
        aug_pipe.fit(X_aug, y_aug)
        aug_pred[test_idx] = aug_pipe.predict(
            np.asarray([base_feats[i] for i in test_idx])
        )
        print(f"  fold {fold}/{n_splits} done")

    base_metrics = evaluate(ye, base_pred)
    aug_metrics = evaluate(ye, aug_pred)
    print("\n" + "=" * 72)
    print("CV COMPARISON")
    print(f"{'':22s} {'baseline':>10s} {'augmented':>10s}")
    print(f"{'accuracy':22s} {base_metrics['accuracy']:>9.1%} "
          f"{aug_metrics['accuracy']:>10.1%}")
    print(f"{'balanced accuracy':22s} "
          f"{base_metrics['balanced_accuracy']:>9.1%} "
          f"{aug_metrics['balanced_accuracy']:>10.1%}")

    labels_idx = list(range(len(le.classes_)))
    base_recall = recall_score(ye, base_pred, labels=labels_idx,
                               average=None, zero_division=0)
    aug_recall = recall_score(ye, aug_pred, labels=labels_idx,
                              average=None, zero_division=0)
    print(f"\n{'class':18s} {'support':>8s} {'base R':>8s} {'aug R':>8s}")
    for i, name in enumerate(le.classes_):
        support = int((ye == i).sum())
        print(f"{name:18s} {support:>8d} {base_recall[i]:>7.0%} "
              f"{aug_recall[i]:>8.0%}")

    print("\nBaseline report:")
    print(classification_report(ye, base_pred, target_names=le.classes_,
                                zero_division=0))
    print("Augmented report:")
    print(classification_report(ye, aug_pred, target_names=le.classes_,
                                zero_division=0))

    print("[5/5] Saving augmented model...")
    X_all, y_all = expand_train(np.arange(len(ye)), base_feats, ye, variants)
    final = make_pipeline(y_all, args.sampler)
    final.fit(X_all, y_all)
    os.makedirs(args.output_dir, exist_ok=True)
    joblib.dump(final, os.path.join(args.output_dir, "cry_classifier.joblib"))
    joblib.dump(le, os.path.join(args.output_dir, "label_encoder.joblib"))
    Xs_all = final.named_steps["scaler"].transform(np.asarray(X_all))
    np.savez_compressed(
        os.path.join(args.output_dir, "source_stats.npz"),
        mean=Xs_all.mean(axis=0).astype(np.float32),
        cov=np.cov(Xs_all, rowvar=False).astype(np.float32),
    )
    report = {
        "training": "donate+dbl+dynamic + minority augmentation",
        "feature_loudness": loudness,
        "file_cmvn": bool(args.file_cmvn),
        "cmvn_min_segments": 2,
        "cmvn_window": 8,
        "channel_augment": {
            "enabled": bool(args.channel_augment),
            "copies": int(args.channel_copies),
            "transforms": ["spectral_eq", "bandlimit", "reverb", "soft_clip"],
        },
        "augmentation": {
            "target_per_class": args.target_per_class,
            "max_per_source": args.max_per_source,
            "generated": dict(generated),
            "transforms": ["pitch_shift", "time_stretch", "gain", "noise"],
        },
        "cv_folds": n_splits,
        "sampler": args.sampler,
        "baseline": {
            "accuracy": round(base_metrics["accuracy"], 4),
            "balanced_accuracy": round(base_metrics["balanced_accuracy"], 4),
            "per_class_recall": {
                name: round(float(base_recall[i]), 4)
                for i, name in enumerate(le.classes_)
            },
        },
        "augmented": {
            "accuracy": round(aug_metrics["accuracy"], 4),
            "balanced_accuracy": round(aug_metrics["balanced_accuracy"], 4),
            "per_class_recall": {
                name: round(float(aug_recall[i]), 4)
                for i, name in enumerate(le.classes_)
            },
        },
        "n_segments": len(raw),
        "n_features": int(np.asarray(base_feats).shape[1]),
        "classes": list(le.classes_),
    }
    with open(os.path.join(args.output_dir, "model_config.json"), "w") as f:
        json.dump(report, f, indent=2)

    if args.promote:
        promote_model(args.output_dir)
    else:
        print(f"  saved to {args.output_dir}")
        print("  run again with --promote to replace the production model")

    print(f"\nDone in {time.time() - t0:.0f}s")
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-per-class", type=int, default=300)
    parser.add_argument("--max-per-source", type=int, default=40)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--jobs", type=int, default=0)
    parser.add_argument("--rms-loudness", action="store_true",
                        help="normalize loudness by RMS instead of peak")
    parser.add_argument("--file-cmvn", action="store_true",
                        help="subtract each recording's mean feature vector")
    parser.add_argument("--channel-augment", action="store_true",
                        help="add channel/EQ/reverb/saturation training copies")
    parser.add_argument("--channel-copies", type=int, default=1)
    parser.add_argument("--sampler", choices=["svmsmote", "smote", "none"],
                        default="svmsmote")
    parser.add_argument("--output-dir", default=OUTPUT_DIR)
    parser.add_argument("--extra-data", action="append", default=[],
                        help="extra class-folder dataset (e.g. pilot export)")
    parser.add_argument("--promote", action="store_true")
    parser.add_argument("--promote-only", action="store_true",
                        help="promote an existing output dir without training")
    args = parser.parse_args()
    if args.promote_only:
        promote_model(args.output_dir)
    else:
        run(args)


if __name__ == "__main__":
    main()
