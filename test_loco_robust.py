"""LOCO comparison for channel/gain robustness.

Modes:
  base          peak-normalized features (current model)
  rms           RMS loudness normalization
  cmvn          RMS + per-recording feature mean subtraction
  rms_channel   RMS + channel augmentation in training
  cmvn_channel  RMS + CMVN + channel augmentation in training

Augmented copies are only added to training folds. Evaluation always uses
the clean test features, so the comparison does not leak test audio.

Usage:
  python test_loco_robust.py
  python test_loco_robust.py --rebuild-cache
"""
import argparse
import os
import time
import warnings
from collections import Counter
from concurrent.futures import ProcessPoolExecutor

warnings.filterwarnings("ignore")

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import LabelEncoder

from augment import channel_augment
from feature_norm import file_cmvn
from test_loco import (
    CLASSES,
    CORPORA,
    evaluate,
    make_eval_pipeline,
)
from train_augmented import load_raw
from train_combined_v2 import SR, extract_features

CACHE = os.path.expanduser("~/.cache/cry-agent/loco_robust_features_v1.npz")
REPORT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "reports", "loco_robust.md"
)


def _extract_one(payload):
    y, loudness, channel = payload
    if channel:
        y = channel_augment(y, SR, seed=channel)
    return extract_features(y, SR, loudness=loudness)


def featurize(raw, loudness, jobs, channel=False):
    payload = [
        (y, loudness, (i * 104729 + 17) if channel else 0)
        for i, y in enumerate(raw)
    ]
    if jobs <= 1:
        return [_extract_one(p) for p in payload]
    with ProcessPoolExecutor(max_workers=jobs) as pool:
        return list(pool.map(_extract_one, payload, chunksize=8))


def build_banks(rebuild=False, jobs=0):
    if os.path.exists(CACHE) and not rebuild:
        data = np.load(CACHE, allow_pickle=False)
        print(f"  cached features: {CACHE}")
        return (
            data["X_base"], data["X_rms"], data["X_chan"],
            data["y"], data["source"], data["files"],
        )

    print("  loading raw segments...")
    raw, labels, sources, files = load_raw()
    keep = np.isin(labels, CLASSES)
    raw = [seg for seg, flag in zip(raw, keep) if flag]
    labels = np.asarray(labels)[keep]
    sources = np.asarray(sources)[keep]
    files = np.asarray(files)[keep]
    print(f"  segments={len(raw)}")

    jobs = jobs or max(1, (os.cpu_count() or 2) - 1)
    banks = {}
    for name, loudness, channel in (
        ("base", "peak", False),
        ("rms", "rms", False),
        ("chan", "rms", True),
    ):
        t0 = time.time()
        print(f"  extracting {name} ({jobs} workers)...")
        feats = featurize(raw, loudness, jobs, channel)
        banks[name] = np.asarray(feats, dtype=np.float32)
        print(f"    {banks[name].shape} in {time.time() - t0:.0f}s")

    os.makedirs(os.path.dirname(CACHE), exist_ok=True)
    np.savez_compressed(
        CACHE,
        X_base=banks["base"], X_rms=banks["rms"], X_chan=banks["chan"],
        y=labels, source=sources, files=files,
    )
    print(f"  cached -> {CACHE}")
    return (
        banks["base"], banks["rms"], banks["chan"],
        labels, sources, files,
    )


def run_mode(mode, X_base, X_rms, X_chan, y, sources, files, splits, le):
    if mode == "base":
        X, A = X_base, None
    else:
        X, A = X_rms, None
        if "channel" in mode:
            A = X_chan

    if mode.startswith("cmvn"):
        X, n_norm = file_cmvn(X, files)
        if A is not None:
            A, _ = file_cmvn(A, files)
        print(f"  {mode}: CMVN applied to {n_norm}/{len(X)} segments")

    def train_matrix(train_idx):
        parts = [X[train_idx]]
        labels = [y[train_idx]]
        if A is not None:
            parts.append(A[train_idx])
            labels.append(y[train_idx])
        return np.vstack(parts), np.concatenate(labels)

    y_enc = le.transform(y)
    group_pred = np.zeros(len(y), dtype=int)
    group_majority = np.zeros(len(y), dtype=int)
    for train_idx, test_idx in splits:
        X_tr, y_tr = train_matrix(train_idx)
        pipe = make_eval_pipeline(le.transform(y_tr))
        pipe.fit(X_tr, le.transform(y_tr))
        group_pred[test_idx] = pipe.predict(X[test_idx])
        group_majority[test_idx] = int(
            np.bincount(le.transform(y_tr)).argmax()
        )
    results = [evaluate(
        f"{mode} group5", y_enc, group_pred, len(CLASSES),
        majority_pred=group_majority,
    )]

    for held_out in CORPORA:
        train_idx = np.where(sources != held_out)[0]
        test_idx = np.where(sources == held_out)[0]
        X_tr, y_tr = train_matrix(train_idx)
        pipe = make_eval_pipeline(le.transform(y_tr))
        pipe.fit(X_tr, le.transform(y_tr))
        results.append(evaluate(
            f"{mode} LOCO {held_out}", y_enc[test_idx],
            pipe.predict(X[test_idx]), len(CLASSES),
            y_train=le.transform(y_tr),
        ))
    return results


def cmvn_coverage(files, sources):
    groups = {}
    for idx, path in enumerate(files):
        groups.setdefault(path, []).append(idx)
    out = {}
    for corpus in CORPORA:
        idx = np.where(sources == corpus)[0]
        covered = sum(1 for i in idx if len(groups[files[i]]) >= 2)
        out[corpus] = (covered, len(idx))
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rebuild-cache", action="store_true")
    parser.add_argument("--jobs", type=int, default=0)
    parser.add_argument("--out", default=REPORT)
    args = parser.parse_args()

    t0 = time.time()
    print("=" * 72)
    print("LOCO ROBUSTNESS: channel/gain normalization + augmentation")
    print("=" * 72)
    print("[1/3] Building feature banks...")
    X_base, X_rms, X_chan, y, sources, files = build_banks(
        args.rebuild_cache, args.jobs
    )
    le = LabelEncoder().fit(CLASSES)
    y_enc = le.transform(y)
    print(f"  classes: {CLASSES}")
    print(f"  per class: {dict(Counter(y))}")
    print(f"  per corpus: {dict(Counter(sources))}")

    print("\n[2/3] Splits...")
    splits = list(
        StratifiedGroupKFold(
            n_splits=5, shuffle=True, random_state=42
        ).split(X_base, y_enc, groups=files)
    )

    print("\n[3/3] Modes...")
    modes = ["base", "rms", "cmvn", "rms_channel", "cmvn_channel"]
    runs = {}
    for mode in modes:
        print(f"\n--- {mode}")
        runs[mode] = run_mode(
            mode, X_base, X_rms, X_chan, y, sources, files, splits, le
        )

    base = runs["base"]
    base_mean_acc = 100 * float(np.mean([r["accuracy"] for r in base[1:]]))
    base_mean_bal = 100 * float(
        np.mean([r["balanced_accuracy"] for r in base[1:]])
    )

    lines = [
        "# LOCO: channel/gain robustness",
        "",
        "Channel augmentation = random smooth EQ, soft band limit, short",
        "reverb, and gain + soft saturation. Augmented copies are training-",
        "only. CMVN subtracts each recording's mean feature vector; files",
        "with a single segment are left unchanged.",
        "",
        "| Mode | Group acc / bal | Donate | DBL | Dynamic | LOCO mean | vs base |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for mode in modes:
        results = runs[mode]
        group = results[0]
        loco = results[1:]
        mean_acc = 100 * float(np.mean([r["accuracy"] for r in loco]))
        mean_bal = 100 * float(
            np.mean([r["balanced_accuracy"] for r in loco])
        )
        cells = [
            f"{r['accuracy']:.1%} / {r['balanced_accuracy']:.1%}"
            for r in loco
        ]
        lines.append(
            f"| {mode} | {group['accuracy']:.1%} / "
            f"{group['balanced_accuracy']:.1%} | "
            + " | ".join(cells)
            + f" | **{mean_acc:.1f}% / {mean_bal:.1f}%** | "
            f"{mean_acc - base_mean_acc:+.1f} / "
            f"{mean_bal - base_mean_bal:+.1f} |"
        )

    total_n = sum(r["n"] for r in base[1:])
    majority = sum(
        r["n"] * r["majority_accuracy"] for r in base[1:]
    ) / max(total_n, 1)
    coverage = cmvn_coverage(files, sources)
    lines += [
        "",
        f"Weighted majority baseline across LOCO corpora: {majority:.1%}.",
        "",
        "CMVN coverage (segments in recordings with >= 2 segments): "
        + ", ".join(
            f"{k} {n}/{total} ({n / total:.1%})"
            for k, (n, total) in coverage.items()
        )
        + ".",
        "",
        "Caveats: CMVN and channel banks are generated once over the whole",
        "dataset, but copies are only used for training indices in each fold.",
        "CMVN uses each recording's own unlabeled segments (transductive);",
        "a live deployment needs a rolling session buffer.",
    ]
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n" + "\n".join(lines))
    print(f"\nSaved {args.out} ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
