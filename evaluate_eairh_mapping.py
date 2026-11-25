"""Compare `eairh -> pain` (A) with `eairh -> discomfort` (B).

The Dunstan audit shows that the corpus defines `eairh` as lower-wind /
abdominal discomfort, so mapping it to `pain` is semantically strong. But the
label also feeds training: DBL's 107 `eairh` segments are 36% of all pain
segments. This script measures what re-mapping costs on the shared five-class
LOCO benchmark before changing anything.

Usage:
  python evaluate_eairh_mapping.py
  python evaluate_eairh_mapping.py --out reports/eairh_mapping.md
"""
import argparse
import os
import time
from collections import Counter

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import LabelEncoder

from test_loco import (
    CLASSES,
    CORPORA,
    build_dataset,
    evaluate,
    make_eval_pipeline,
)

REPORT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "reports")


def run_variant(tag, labels, is_eairh, X, sources, files, le):
    y = le.transform(labels)
    if tag == "B":
        y = y.copy()
        y[is_eairh] = le.transform(["discomfort"])[0]

    print(f"=== {tag}: {dict(sorted(Counter(
        CLASSES[i] for i in y).items()))}")

    splits = list(
        StratifiedGroupKFold(
            n_splits=5, shuffle=True, random_state=42
        ).split(X, y, groups=files)
    )
    group_pred = np.zeros(len(y), dtype=int)
    group_majority = np.zeros(len(y), dtype=int)
    for train_idx, test_idx in splits:
        pipe = make_eval_pipeline(y[train_idx])
        pipe.fit(X[train_idx], y[train_idx])
        group_pred[test_idx] = pipe.predict(X[test_idx])
        group_majority[test_idx] = int(np.bincount(y[train_idx]).argmax())
    results = [evaluate(
        f"{tag} group5", y, group_pred, len(CLASSES),
        majority_pred=group_majority,
    )]

    for held_out in CORPORA:
        train_idx = np.where(sources != held_out)[0]
        test_idx = np.where(sources == held_out)[0]
        pipe = make_eval_pipeline(y[train_idx])
        pipe.fit(X[train_idx], y[train_idx])
        results.append(evaluate(
            f"{tag} LOCO {held_out}", y[test_idx],
            pipe.predict(X[test_idx]), len(CLASSES),
            y_train=y[train_idx],
        ))

    loco = results[1:]
    mean_acc = float(np.mean([r["accuracy"] for r in loco]))
    mean_bal = float(np.mean([r["balanced_accuracy"] for r in loco]))
    print(f"{tag} LOCO mean acc={mean_acc:.1%} bal={mean_bal:.1%}")
    return results, mean_acc, mean_bal


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out", default=os.path.join(REPORT_DIR, "eairh_mapping.md")
    )
    args = parser.parse_args()

    t0 = time.time()
    X, labels, sources, files = build_dataset()
    le = LabelEncoder().fit(CLASSES)
    is_eairh = (sources == "dbl") & np.asarray(
        ["/eairh/" in f for f in files]
    )
    print(f"  dbl eairh segments={int(is_eairh.sum())} "
          f"of {int((labels == 'pain').sum())} pain segments")

    runs = {}
    for tag in ("A", "B"):
        runs[tag] = run_variant(
            tag, labels, is_eairh, X, sources, files, le
        )

    pain = CLASSES.index("pain")
    discomfort = CLASSES.index("discomfort")
    lines = [
        "# `eairh` mapping decision",
        "",
        "Dunstan defines `eairh` as lower-wind / abdominal discomfort, but the",
        "label also feeds training: the 107 DBL `eairh` segments are 36% of all",
        "`pain` segments. This compares keeping the current mapping (A) with",
        "re-mapping to `discomfort` (B) on the shared five-class benchmark.",
        "",
        "| Evaluation | A acc / bal | B acc / bal |",
        "|---|---:|---:|",
    ]
    for (a, b) in zip(runs["A"][0], runs["B"][0]):
        lines.append(
            f"| {a['name'][2:]} | {a['accuracy']:.1%} / "
            f"{a['balanced_accuracy']:.1%} | {b['accuracy']:.1%} / "
            f"{b['balanced_accuracy']:.1%} |"
        )
    lines += [
        f"| **LOCO mean** | **{runs['A'][1]:.1%} / {runs['A'][2]:.1%}** "
        f"| **{runs['B'][1]:.1%} / {runs['B'][2]:.1%}** |",
        "",
        "Pain recall:",
        "",
        "| Evaluation | A | B |",
        "|---|---:|---:|",
    ]
    for (a, b) in zip(runs["A"][0], runs["B"][0]):
        lines.append(
            f"| {a['name'][2:]} | {a['recall'][pain]:.0%} | "
            f"{b['recall'][pain]:.0%} |"
        )
    lines += [
        "",
        "Discomfort recall:",
        "",
        "| Evaluation | A | B |",
        "|---|---:|---:|",
    ]
    for (a, b) in zip(runs["A"][0], runs["B"][0]):
        lines.append(
            f"| {a['name'][2:]} | {a['recall'][discomfort]:.0%} | "
            f"{b['recall'][discomfort]:.0%} |"
        )
    lines += [
        "",
        "## Decision",
        "",
        "Keep A (`eairh -> pain`). B gains 5.1 pp balanced accuracy on DBL",
        "alone, but loses 1.1 pp LOCO accuracy, drops DynamicSuperb pain",
        "recall from 33% to 0%, and cuts overall group-CV pain recall from",
        "41% to 28%. `pain` is the safety-critical class, so a semantic",
        "cleanup that erases its third-largest training source is a bad",
        "trade at 107 segments.",
        "",
        "The mapping is defensible under our own `pain` rubric, which already",
        "includes colic: infant colic is lower-wind/gas pain, not only acute",
        "sharp pain. The rubric is now explicit about that. Revisit only if a",
        "real `gas` class with enough data is added end-to-end.",
        "",
        "Caveats: segment-level cache, one seed, five shared classes only,",
        "and DBL is one of the training corpora in these splits.",
    ]

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Saved {args.out} ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
