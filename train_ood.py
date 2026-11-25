"""Train and evaluate the out-of-distribution gate.

The gate is a per-class Mahalanobis distance in standardized audio-feature
space. Threshold calibration uses GroupKFold by recording, so it reflects
unseen recordings rather than training points.

Usage:
  python train_ood.py
  python train_ood.py --percentile 85 --rebuild-cache
"""
import argparse
import json
import os
import time
import warnings
from collections import Counter

warnings.filterwarnings("ignore")

import joblib
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder

from ood import OODDetector
from feature_norm import file_cmvn
from test_loco import make_eval_pipeline
from train_augmented import featurize, load_raw

CACHE = os.path.expanduser("~/.cache/cry-agent/ood_features_v1.npz")
LOCO_CACHE = os.path.expanduser("~/.cache/cry-agent/loco_features_v1.npz")
ROBUST_CACHE = os.path.expanduser(
    "~/.cache/cry-agent/loco_robust_features_v1.npz"
)
MODEL_PATH = os.path.expanduser(
    "~/.config/cry-agent/models/ood_detector.joblib"
)
REPORT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "reports")
LOCO_CLASSES = ["burp", "discomfort", "hungry", "pain", "sleepy"]
LOCO_CORPORA = ("donate", "dbl", "dynamic")


def build_features(rebuild=False, jobs=0):
    if os.path.exists(CACHE) and not rebuild:
        data = np.load(CACHE, allow_pickle=False)
        print(f"  cached features: {CACHE}")
        return data["X"], data["y"], data["source"], data["files"]

    print("  loading all raw segments...")
    raw, labels, sources, files = load_raw()
    jobs = jobs or max(1, (os.cpu_count() or 2) - 1)
    print(f"  segments={len(raw)}, extracting features ({jobs} workers)...")
    feats = featurize(raw, jobs)
    X = np.asarray(feats, dtype=np.float32)
    os.makedirs(os.path.dirname(CACHE), exist_ok=True)
    np.savez_compressed(
        CACHE, X=X, y=np.asarray(labels),
        source=np.asarray(sources), files=np.asarray(files),
    )
    print(f"  cached -> {CACHE}")
    return X, np.asarray(labels), np.asarray(sources), np.asarray(files)


def evaluate_loco(percentile):
    if not os.path.exists(LOCO_CACHE):
        print(f"  LOCO cache missing: {LOCO_CACHE}")
        return None
    data = np.load(LOCO_CACHE, allow_pickle=False)
    X, y, source, files = (
        data["X"], data["y"], data["source"], data["files"]
    )
    le = LabelEncoder().fit(LOCO_CLASSES)
    ye = le.transform(y)

    results = []
    for held in LOCO_CORPORA:
        train_idx = np.where(source != held)[0]
        test_idx = np.where(source == held)[0]
        detector = OODDetector(
            calibration_percentile=percentile
        ).fit(X[train_idx], ye[train_idx], groups=files[train_idx])
        in_scores = detector.calibration_scores_
        ood_scores = detector.score(X[test_idx])
        auc = roc_auc_score(
            np.r_[np.zeros(len(in_scores)), np.ones(len(ood_scores))],
            np.r_[in_scores, ood_scores],
        )
        tpr = {
            p: float((ood_scores > np.percentile(in_scores, p)).mean())
            for p in (75, 80, 85, 90)
        }
        classifier = make_eval_pipeline(ye[train_idx])
        classifier.fit(X[train_idx], ye[train_idx])
        pred = classifier.predict(X[test_idx])
        ood_flags = detector.is_ood(X[test_idx])
        trusted = ~ood_flags
        trusted_acc = (
            float((pred[trusted] == ye[test_idx][trusted]).mean())
            if trusted.any() else None
        )
        trusted_acc_text = (
            "n/a" if trusted_acc is None else f"{trusted_acc:.0%}"
        )
        results.append({
            "corpus": held,
            "n": int(len(test_idx)),
            "auc": float(auc),
            "tpr": tpr,
            "threshold": float(np.percentile(in_scores, percentile)),
            "in_distribution_p95": float(np.percentile(in_scores, 95)),
            "gate_rate": float(ood_flags.mean()),
            "trusted_n": int(trusted.sum()),
            "trusted_accuracy": trusted_acc,
            "unsafe_rate": (
                float(1.0 - trusted_acc) if trusted_acc is not None else None
            ),
        })
        print(f"  LOCO {held:8s} n={len(test_idx):4d} AUC={auc:.3f} "
              f"TPR@75={tpr[75]:.0%} @80={tpr[80]:.0%} "
              f"@85={tpr[85]:.0%} @90={tpr[90]:.0%} "
              f"gate={ood_flags.mean():.0%} trusted_n={trusted.sum()} "
              f"trusted_acc={trusted_acc_text}")
    return results


def build_robust_features(ood_cmvn=False):
    if not os.path.exists(ROBUST_CACHE):
        raise FileNotFoundError(
            f"robust feature cache missing: {ROBUST_CACHE}. "
            "Run test_loco_robust.py once to build it."
        )
    data = np.load(ROBUST_CACHE, allow_pickle=False)
    X_raw, files = data["X_rms"], data["files"]
    X_cls, covered = file_cmvn(X_raw, files)
    X_ood = X_cls if ood_cmvn else X_raw
    print(f"  classifier CMVN applied to {covered}/{len(X_raw)} segments")
    print(f"  OOD features: {'CMVN' if ood_cmvn else 'raw RMS'}")
    return X_cls, X_ood, data["y"], data["source"], files


def evaluate_loco_robust(percentile, ood_cmvn=False):
    X_cls, X_ood, y, source, files = build_robust_features(ood_cmvn)
    le = LabelEncoder().fit(LOCO_CLASSES)
    ye = le.transform(y)

    results = []
    for held in LOCO_CORPORA:
        train_idx = np.where(source != held)[0]
        test_idx = np.where(source == held)[0]
        detector = OODDetector(
            calibration_percentile=percentile
        ).fit(X_ood[train_idx], ye[train_idx], groups=files[train_idx])
        in_scores = detector.calibration_scores_
        ood_scores = detector.score(X_ood[test_idx])
        auc = roc_auc_score(
            np.r_[np.zeros(len(in_scores)), np.ones(len(ood_scores))],
            np.r_[in_scores, ood_scores],
        )
        tpr = {
            p: float((ood_scores > np.percentile(in_scores, p)).mean())
            for p in (75, 80, 85, 90)
        }
        classifier = make_eval_pipeline(ye[train_idx])
        classifier.fit(X_cls[train_idx], ye[train_idx])
        pred = classifier.predict(X_cls[test_idx])
        ood_flags = detector.is_ood(X_ood[test_idx])
        trusted = ~ood_flags
        trusted_acc = (
            float((pred[trusted] == ye[test_idx][trusted]).mean())
            if trusted.any() else None
        )
        trusted_acc_text = (
            "n/a" if trusted_acc is None else f"{trusted_acc:.0%}"
        )
        results.append({
            "corpus": held,
            "n": int(len(test_idx)),
            "auc": float(auc),
            "tpr": tpr,
            "threshold": float(np.percentile(in_scores, percentile)),
            "in_distribution_p95": float(np.percentile(in_scores, 95)),
            "gate_rate": float(ood_flags.mean()),
            "trusted_n": int(trusted.sum()),
            "trusted_accuracy": trusted_acc,
        })
        print(f"  LOCO {held:8s} n={len(test_idx):4d} AUC={auc:.3f} "
              f"TPR@75={tpr[75]:.0%} @80={tpr[80]:.0%} "
              f"@85={tpr[85]:.0%} @90={tpr[90]:.0%} "
              f"gate={ood_flags.mean():.0%} trusted_n={trusted.sum()} "
              f"trusted_acc={trusted_acc_text}")
    return results


def write_robust_report(production, loco, percentile, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    payload = {
        "production": production,
        "loco": loco,
        "calibration_percentile": percentile,
    }
    json_path = os.path.join(out_dir, "ood_gate_robust.json")
    with open(json_path, "w") as f:
        json.dump(payload, f, indent=2)

    lines = [
        "# OOD gate report — robust feature bank",
        "",
        f"Date: {time.strftime('%Y-%m-%d')}",
        "",
        "Classifier features: RMS loudness + per-recording CMVN.",
        "OOD features: "
        + ("CMVN." if production.get("ood_cmvn") else "raw RMS, no CMVN."),
        "",
        "- Score: minimum per-class Mahalanobis distance after standardization",
        "- Calibration: GroupKFold by recording (unseen recordings)",
        f"- Default threshold: {percentile:.0f}th percentile of in-distribution",
        "  calibration scores",
        f"- Production detector: {production['n_samples']} segments, "
        f"{production['n_classes']} classes",
        f"- Threshold: {production['threshold']:.3f}",
        "",
        "## LOCO detection",
        "",
        "| Held-out corpus | n | AUC | TPR@10% FPR | TPR@15% FPR | "
        "Gate rate | Trusted n | Trusted acc |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    if loco:
        for row in loco:
            trusted_acc_text = (
                "n/a" if row["trusted_accuracy"] is None
                else f"{row['trusted_accuracy']:.0%}"
            )
            lines.append(
                f"| {row['corpus']} | {row['n']} | {row['auc']:.3f} | "
                f"{row['tpr'][90]:.0%} | {row['tpr'][85]:.0%} | "
                f"{row['gate_rate']:.0%} | {row['trusted_n']} | "
                f"{trusted_acc_text} |"
            )
        mean_auc = float(np.mean([r["auc"] for r in loco]))
        mean_tpr90 = float(np.mean([r["tpr"][90] for r in loco]))
        mean_tpr85 = float(np.mean([r["tpr"][85] for r in loco]))
        mean_gate = float(np.mean([r["gate_rate"] for r in loco]))
        trusted_n = sum(r["trusted_n"] for r in loco)
        lines.append(
            f"| **mean** | | **{mean_auc:.3f}** | **{mean_tpr90:.0%}** | "
            f"**{mean_tpr85:.0%}** | **{mean_gate:.0%}** | **{trusted_n}** | |"
        )
    lines += [
        "",
        "## Interpretation",
        "",
        "- This gate is paired with the channel-robust classifier but scores",
        "  OOD on raw RMS features. CMVN removes the recording-level offset",
        "  that the gate relies on; scoring CMVN features drops mean LOCO",
        "  AUC from ~0.95 to ~0.70.",
        "- Do not mix this detector with the peak-normalized production model.",
        "- CMVN is transductive within a recording. Live inference needs two",
        "  segments before the rolling buffer is active.",
        "- A high gate rate on a new corpus is expected: CMVN recovers part of",
        "  the channel shift, not the label/recording-protocol shift.",
        "",
    ]
    md_path = os.path.join(out_dir, "ood_gate_robust.md")
    with open(md_path, "w") as f:
        f.write("\n".join(lines))
    print(f"  saved {json_path}")
    print(f"  saved {md_path}")


def write_report(production, loco, percentile, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    json_path = os.path.join(out_dir, "ood_gate.json")
    payload = {
        "production": production,
        "loco": loco,
        "calibration_percentile": percentile,
    }
    with open(json_path, "w") as f:
        json.dump(payload, f, indent=2)

    lines = [
        "# OOD gate report",
        "",
        f"Date: {time.strftime('%Y-%m-%d')}",
        "",
        "## Method",
        "",
        "- Score: minimum per-class Mahalanobis distance after standardization",
        "- Calibration: GroupKFold by recording (unseen recordings)",
        f"- Default threshold: {percentile:.0f}th percentile of in-distribution",
        "  calibration scores",
        f"- Production detector: {production['n_samples']} segments, "
        f"{production['n_classes']} classes",
        f"- Threshold: {production['threshold']:.3f}",
        "",
        "## LOCO detection",
        "",
        "| Held-out corpus | n | AUC | TPR@10% FPR | TPR@15% FPR | "
        "Gate rate | Trusted n | Trusted acc |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    if loco:
        for row in loco:
            trusted_acc_text = (
                "n/a" if row["trusted_accuracy"] is None
                else f"{row['trusted_accuracy']:.0%}"
            )
            lines.append(
                f"| {row['corpus']} | {row['n']} | {row['auc']:.3f} | "
                f"{row['tpr'][90]:.0%} | {row['tpr'][85]:.0%} | "
                f"{row['gate_rate']:.0%} | {row['trusted_n']} | "
                f"{trusted_acc_text} |"
            )
        mean_auc = float(np.mean([r["auc"] for r in loco]))
        mean_tpr90 = float(np.mean([r["tpr"][90] for r in loco]))
        mean_tpr85 = float(np.mean([r["tpr"][85] for r in loco]))
        mean_gate = float(np.mean([r["gate_rate"] for r in loco]))
        trusted_n = sum(r["trusted_n"] for r in loco)
        lines.append(
            f"| **mean** | | **{mean_auc:.3f}** | **{mean_tpr90:.0%}** | "
            f"**{mean_tpr85:.0%}** | **{mean_gate:.0%}** | **{trusted_n}** | |"
        )
    lines += [
        "",
        "## Interpretation",
        "",
        "- The gate is a safety mechanism, not a classifier. It flags inputs",
        "  that are far from every training class and turns them into",
        "  `unknown` / low-trust decisions.",
        "- At 15% false-alarm rate it catches about 90% of held-out corpus",
        "  segments on average; DynamicSuperb is the hardest corpus.",
        "- The few held-out segments that pass the gate are still only",
        "  16-19% accurate on DBL/DynamicSuperb. Per-sample gating alone is",
        "  not enough; the pipeline also has a session-level gate (8 of the",
        "  last 12 clips must exceed the threshold to mark the session OOD).",
        "- A gate cannot recover the lost information. It buys honesty, not",
        "  accuracy. Domain adaptation and family data are still required.",
        "- Do not interpret 'not flagged' as 'trustworthy' on a new corpus.",
        "",
    ]
    md_path = os.path.join(out_dir, "ood_gate.md")
    with open(md_path, "w") as f:
        f.write("\n".join(lines))
    print(f"  saved {json_path}")
    print(f"  saved {md_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--percentile", type=float, default=85.0)
    parser.add_argument("--rebuild-cache", action="store_true")
    parser.add_argument("--jobs", type=int, default=0)
    parser.add_argument("--out", default=MODEL_PATH)
    parser.add_argument("--report-dir", default=REPORT_DIR)
    parser.add_argument("--robust", action="store_true",
                        help="pair with the robust model: classify on CMVN, "
                             "score OOD on RMS")
    parser.add_argument("--robust-cmvn-ood", action="store_true",
                        help="score OOD on CMVN features too (default: raw RMS)")
    args = parser.parse_args()

    t0 = time.time()
    print("=" * 72)
    print("OOD GATE TRAINING")
    print("=" * 72)
    print("[1/3] Loading features...")
    if args.robust:
        X_cls, X_ood, y, sources, files = build_robust_features(
            args.robust_cmvn_ood
        )
        if args.out == MODEL_PATH:
            args.out = os.path.expanduser(
                "~/.config/cry-agent/models_robust/ood_detector.joblib"
            )
    else:
        X, y, sources, files = build_features(args.rebuild_cache, args.jobs)
        X_ood = None
    fit_X = X_ood if args.robust else X
    print(f"  segments={len(fit_X)} classes={dict(Counter(y.tolist()))}")

    print("\n[2/3] Fitting production detector...")
    detector = OODDetector(
        calibration_percentile=args.percentile
    ).fit(fit_X, y, groups=files)
    if args.robust:
        detector.metadata["feature_bank"] = "rms"
        detector.metadata["ood_cmvn"] = bool(args.robust_cmvn_ood)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    joblib.dump(detector, args.out)
    if args.robust:
        config_path = os.path.join(
            os.path.dirname(args.out), "model_config.json"
        )
        if os.path.exists(config_path):
            with open(config_path) as f:
                config = json.load(f)
            config["ood_on_raw_features"] = not args.robust_cmvn_ood
            with open(config_path, "w") as f:
                json.dump(config, f, indent=2)
            print(f"  updated {config_path}")
    print(f"  {detector.metadata}")
    print(f"  saved {args.out}")

    print("\n[3/3] LOCO detection evaluation...")
    if args.robust:
        loco = evaluate_loco_robust(
            args.percentile, args.robust_cmvn_ood
        )
        write_robust_report(
            detector.metadata, loco, args.percentile, args.report_dir
        )
    else:
        loco = evaluate_loco(args.percentile)
        write_report(detector.metadata, loco, args.percentile, args.report_dir)
    print(f"\nDone in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
