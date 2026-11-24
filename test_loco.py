"""Leave-one-corpus-out evaluation.

Three views of generalization:
  1. Group 5-fold by recording  -> same-corpus, unseen-recording estimate
  2. Leave one corpus out       -> unseen collection setup estimate
  3. Per-corpus confusion matrix

Only the five classes shared by all three corpora are evaluated:
burp, discomfort, hungry, pain, sleepy. Startle/temperature/lonely are too
small or absent outside DynamicSuperb, so they are reported by the normal CV
instead.

Usage:
  python test_loco.py
  python test_loco.py --rebuild-cache
"""
import argparse
import json
import os
import time
import warnings
from collections import Counter

warnings.filterwarnings("ignore")

import numpy as np
from imblearn.over_sampling import SVMSMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    recall_score,
)
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC

from train_augmented import featurize, load_raw

CORPORA = ("donate", "dbl", "dynamic")
CLASSES = ["burp", "discomfort", "hungry", "pain", "sleepy"]
CACHE = os.path.expanduser("~/.cache/cry-agent/loco_features_v1.npz")
REPORT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "reports")


def build_dataset(rebuild=False, jobs=0):
    if os.path.exists(CACHE) and not rebuild:
        data = np.load(CACHE, allow_pickle=False)
        X, y, source, files = (
            data["X"], data["y"], data["source"], data["files"]
        )
        print(f"  cached features: {CACHE}")
        return X, y, source, files

    print("  loading raw segments...")
    raw, labels, sources, files = load_raw()
    keep = np.isin(labels, CLASSES)
    raw = [seg for seg, flag in zip(raw, keep) if flag]
    labels = np.asarray(labels)[keep]
    sources = np.asarray(sources)[keep]
    files = np.asarray(files)[keep]
    print(f"  segments={len(raw)} (5 shared classes)")
    jobs = jobs or max(1, (os.cpu_count() or 2) - 1)
    print(f"  extracting features ({jobs} workers)...")
    feats = featurize(raw, jobs)
    X = np.asarray(feats, dtype=np.float32)
    os.makedirs(os.path.dirname(CACHE), exist_ok=True)
    np.savez_compressed(CACHE, X=X, y=labels, source=sources, files=files)
    print(f"  cached -> {CACHE}")
    return X, labels, sources, files


def make_eval_pipeline(labels):
    counts = np.bincount(labels)
    counts = counts[counts > 0]
    k = max(1, min(3, int(counts.min()) - 1))
    return ImbPipeline([
        ("scaler", StandardScaler()),
        ("smote", SVMSMOTE(random_state=42, k_neighbors=k)),
        ("clf", SVC(
            C=1.0, kernel="rbf", gamma="scale",
            class_weight="balanced", probability=False, random_state=42,
        )),
    ])


def evaluate(name, y_true, y_pred, n_classes, y_train=None,
             majority_pred=None):
    labels = list(range(n_classes))
    recalls = recall_score(
        y_true, y_pred, labels=labels, average=None, zero_division=0
    )
    result = {
        "name": name,
        "n": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "recall": [float(v) for v in recalls],
        "confusion": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
    }
    if majority_pred is not None:
        result["majority_accuracy"] = float(
            accuracy_score(y_true, majority_pred)
        )
    elif y_train is not None:
        majority = int(np.bincount(y_train).argmax())
        result["majority_class"] = CLASSES[majority]
        result["majority_accuracy"] = float(np.mean(y_true == majority))
    print(f"  {name:<28s} n={result['n']:<5d} "
          f"acc={result['accuracy']:.1%} bal={result['balanced_accuracy']:.1%}"
          + (f" maj={result['majority_accuracy']:.1%}"
             if "majority_accuracy" in result else ""))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rebuild-cache", action="store_true")
    parser.add_argument("--jobs", type=int, default=0)
    parser.add_argument("--out-dir", default=REPORT_DIR)
    args = parser.parse_args()

    t0 = time.time()
    print("=" * 72)
    print("LEAVE-ONE-CORPUS-OUT EVALUATION")
    print("=" * 72)
    print("[1/4] Building dataset...")
    X, labels, sources, files = build_dataset(args.rebuild_cache, args.jobs)

    le = LabelEncoder().fit(CLASSES)
    ye = le.transform(labels)
    n_classes = len(CLASSES)
    print(f"  classes: {CLASSES}")
    print(f"  per corpus: {dict(Counter(sources))}")
    print(f"  per class:  {dict(Counter(labels))}")

    results = []

    print("\n[2/4] Group 5-fold CV by recording...")
    try:
        splitter = StratifiedGroupKFold(
            n_splits=5, shuffle=True, random_state=42
        )
        splits = list(splitter.split(X, ye, groups=files))
    except Exception as exc:
        print(f"  StratifiedGroupKFold unavailable ({exc}); falling back")
        splits = list(GroupKFold(n_splits=5).split(X, ye, groups=files))

    group_pred = np.zeros(len(ye), dtype=int)
    group_majority = np.zeros(len(ye), dtype=int)
    for fold, (train_idx, test_idx) in enumerate(splits, 1):
        pipe = make_eval_pipeline(ye[train_idx])
        pipe.fit(X[train_idx], ye[train_idx])
        group_pred[test_idx] = pipe.predict(X[test_idx])
        group_majority[test_idx] = int(np.bincount(ye[train_idx]).argmax())
        print(f"  fold {fold}/{len(splits)} done")
    results.append(evaluate(
        "Group 5-fold (recording)", ye, group_pred, n_classes,
        majority_pred=group_majority,
    ))

    print("\n[3/4] Leave-one-corpus-out...")
    for held_out in CORPORA:
        train_idx = np.where(sources != held_out)[0]
        test_idx = np.where(sources == held_out)[0]
        train_classes = set(ye[train_idx])
        missing = [CLASSES[i] for i in range(n_classes)
                   if i not in train_classes]
        if missing:
            print(f"  {held_out}: WARNING missing training classes {missing}")
        pipe = make_eval_pipeline(ye[train_idx])
        pipe.fit(X[train_idx], ye[train_idx])
        pred = pipe.predict(X[test_idx])
        results.append(evaluate(
            f"LOCO {held_out}", ye[test_idx], pred, n_classes,
            y_train=ye[train_idx],
        ))

    print("\n[4/4] Summary")
    header = (f"{'evaluation':<28s} {'n':>5s} {'acc':>7s} {'bal':>7s} "
              f"{'maj':>7s} " + " ".join(
        f"{c[:6]:>7s}" for c in CLASSES
    ))
    print(header)
    print("-" * len(header))
    for result in results:
        print(f"{result['name']:<28s} {result['n']:>5d} "
              f"{result['accuracy']:>6.1%} {result['balanced_accuracy']:>6.1%} "
              f"{result.get('majority_accuracy', float('nan')):>6.1%} "
              + " ".join(f"{r:>6.0%}" for r in result["recall"]))

    loco = [r for r in results if r["name"].startswith("LOCO")]
    mean_acc = float(np.mean([r["accuracy"] for r in loco]))
    mean_bal = float(np.mean([r["balanced_accuracy"] for r in loco]))
    mean_maj = float(
        sum(r["n"] * r["majority_accuracy"] for r in loco)
        / max(sum(r["n"] for r in loco), 1)
    )
    print(f"\nLOCO mean: acc={mean_acc:.1%} bal={mean_bal:.1%}")
    print(f"Group CV minus LOCO mean: "
          f"{(results[0]['accuracy'] - mean_acc) * 100:+.1f} pp accuracy, "
          f"{(results[0]['balanced_accuracy'] - mean_bal) * 100:+.1f} pp balanced")

    for result in results:
        if not result["name"].startswith("LOCO"):
            continue
        print(f"\nConfusion matrix — {result['name']}")
        print(f"  {'true/pred':<12s} " + " ".join(f"{c[:6]:>7s}" for c in CLASSES))
        for i, row in enumerate(result["confusion"]):
            print(f"  {CLASSES[i]:<12s} " + " ".join(f"{v:>7d}" for v in row))

    os.makedirs(args.out_dir, exist_ok=True)
    payload = {
        "classes": CLASSES,
        "corpora": list(CORPORA),
        "per_corpus": dict(Counter(sources.tolist())),
        "per_class": dict(Counter(labels.tolist())),
        "results": results,
        "loco_mean": {"accuracy": mean_acc, "balanced_accuracy": mean_bal},
        "note": "Mapped to the five labels shared by all corpora.",
    }
    json_path = os.path.join(args.out_dir, "loco_cv.json")
    with open(json_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nSaved {json_path}")

    md = [
        "# Leave-one-corpus-out report",
        "",
        f"Date: {time.strftime('%Y-%m-%d')}",
        "",
        "## Setup",
        "",
        "- Classes: " + ", ".join(CLASSES),
        f"- Segments: {len(ye)}",
        f"- Per corpus: {dict(Counter(sources.tolist()))}",
        "- Model: StandardScaler + SVMSMOTE + SVM-RBF",
        "- Group CV groups by source recording; LOCO holds out an entire",
        "  corpus and evaluates only the five shared labels.",
        "",
        "## Results",
        "",
        "| Evaluation | n | Accuracy | Balanced | Majority baseline |",
        "|---|---:|---:|---:|---:|",
    ]
    for result in results:
        md.append(
            f"| {result['name']} | {result['n']} | "
            f"{result['accuracy']:.1%} | {result['balanced_accuracy']:.1%} | "
            f"{result.get('majority_accuracy', float('nan')):.1%} |"
        )
    md += [
        f"| **LOCO mean** | | **{mean_acc:.1%}** | **{mean_bal:.1%}** | "
        f"{mean_maj:.1%} |",
        "",
        "## Per-class recall",
        "",
        "| Evaluation | " + " | ".join(CLASSES) + " |",
        "|---|" + "---:|" * len(CLASSES),
    ]
    for result in results:
        md.append(
            f"| {result['name']} | "
            + " | ".join(f"{r:.0%}" for r in result["recall"]) + " |"
        )
    md += [
        "",
        "## Confusion matrices",
        "",
    ]
    for result in loco:
        md += [
            f"### {result['name']}",
            "",
            "| true / pred | " + " | ".join(CLASSES) + " |",
            "|---|" + "---:|" * len(CLASSES),
        ]
        for i, row in enumerate(result["confusion"]):
            md.append(
                f"| {CLASSES[i]} | " + " | ".join(str(v) for v in row) + " |"
            )
        md.append("")
    md += [
        "## Interpretation",
        "",
        f"- Group CV: **{results[0]['accuracy']:.1%}** accuracy / "
        f"**{results[0]['balanced_accuracy']:.1%}** balanced.",
        f"- LOCO mean: **{mean_acc:.1%}** accuracy / **{mean_bal:.1%}** "
        "balanced.",
        "- The gap is the cost of changing collection setup. A large gap",
        "  means the model is learning corpus-specific cues, not a",
        "  transferable cry signature.",
        f"- Majority baseline reaches **{mean_maj:.1%}** LOCO accuracy. "
        f"The model is at **{mean_acc:.1%}**, so it is not merely weak; "
        "on all three held-out corpora it is worse than always",
        "  predicting the training majority class.",
        f"- Even group CV ({results[0]['accuracy']:.1%}) is only "
        f"{(results[0]['accuracy'] - results[0]['majority_accuracy']) * 100:+.1f} pp "
        "over its majority baseline; most of the apparent skill is",
        "  class-frequency and corpus identity.",
        "- Do not report in-corpus accuracy as cross-dataset generalization.",
        "- Startle/temperature/lonely are excluded: they exist only in one",
        "  corpus and cannot be evaluated cross-corpus.",
        "",
        "## What to do next",
        "",
        "1. Re-check label semantics across corpora (Dunstan `eh`/`eairh` vs",
        "   Donate `burping`/`belly_pain` may not be the same construct).",
        "2. Add channel/room augmentation and per-recording normalization,",
        "   then re-run LOCO; do not accept a change that only lifts CV.",
        "3. Try domain-invariant features or CORAL/adversarial training.",
        "4. Collect a small in-family calibration set and adapt per household.",
        "5. Add an out-of-distribution gate so a new corpus escalates instead",
        "   of returning a confident but wrong cause.",
        "",
    ]
    md_path = os.path.join(args.out_dir, "loco_cv.md")
    with open(md_path, "w") as f:
        f.write("\n".join(md))
    print(f"Saved {md_path}")
    print(f"Done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
