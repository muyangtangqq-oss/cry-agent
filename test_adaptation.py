"""Few-shot family calibration under leave-one-corpus-out.

Each held-out corpus plays the role of a new family:

  * the global model is trained on the other two corpora
  * a small labelled calibration set is drawn from the held-out corpus
    (split by recording, never by segment)
  * the rest of the held-out corpus is the test set

Methods compared:
  global            - current SVM
  prior             - global + uniform target-prior correction
  proto-k           - k clips/class -> nearest centroid
  blend-k           - 0.5 * global + 0.5 * prototype
  coral             - unsupervised covariance alignment, then global

Usage:
  python test_adaptation.py
  python test_adaptation.py --seeds 5 --shots 3 5 10
"""
import argparse
import json
import os
import time
import warnings
from collections import Counter, defaultdict

warnings.filterwarnings("ignore")

import numpy as np
from imblearn.over_sampling import SVMSMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.metrics import accuracy_score, balanced_accuracy_score, recall_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from adapt import CoralAdapter, PriorCorrector, PrototypeAdapter, blend_proba
from test_loco import CLASSES, CORPORA, REPORT_DIR, build_dataset


def build_global_model(X, y):
    counts = np.bincount(y)
    counts = counts[counts > 0]
    k = max(1, min(3, int(counts.min()) - 1))
    scaler = StandardScaler().fit(X)
    Xs = scaler.transform(X)
    model = ImbPipeline([
        ("smote", SVMSMOTE(random_state=42, k_neighbors=k)),
        ("clf", SVC(
            C=1.0, kernel="rbf", gamma="scale",
            class_weight="balanced", probability=True, random_state=42,
        )),
    ])
    model.fit(Xs, y)
    return scaler, model


def split_recordings(files, y, pool_idx, cal_fraction, seed):
    """Split pool segments into calibration/test by recording."""
    rng = np.random.default_rng(seed)
    by_class = defaultdict(set)
    for idx in pool_idx:
        by_class[int(y[idx])].add(str(files[idx]))

    cal_files, test_files = set(), set()
    for cls, names in by_class.items():
        names = sorted(names)
        rng.shuffle(names)
        n_cal = max(1, int(round(len(names) * cal_fraction)))
        n_cal = min(n_cal, len(names) - 1)  # keep at least one test recording
        cal_files.update(names[:n_cal])
        test_files.update(names[n_cal:])

    cal_idx = [i for i in pool_idx if str(files[i]) in cal_files]
    test_idx = [i for i in pool_idx if str(files[i]) in test_files]
    return np.array(cal_idx), np.array(test_idx)


def sample_shots(cal_idx, y, shots, seed, files):
    """Pick up to `shots` clips per class, preferring distinct recordings."""
    rng = np.random.default_rng(seed)
    picked = []
    for cls in range(len(CLASSES)):
        rows = cal_idx[y[cal_idx] == cls]
        if len(rows) == 0:
            continue
        by_file = defaultdict(list)
        for idx in rows:
            by_file[str(files[idx])].append(int(idx))
        names = sorted(by_file)
        rng.shuffle(names)
        chosen = []
        for name in names:
            if len(chosen) >= shots:
                break
            chosen.append(int(rng.choice(by_file[name])))
        if len(chosen) < shots:
            remaining = [int(i) for i in rows if int(i) not in chosen]
            extra = min(shots - len(chosen), len(remaining))
            if extra:
                chosen.extend(
                    rng.choice(remaining, size=extra, replace=False).tolist()
                )
        picked.extend(chosen)
    return np.array(picked)


def score(y_true, proba):
    pred = np.argmax(proba, axis=1)
    return {
        "accuracy": float(accuracy_score(y_true, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, pred)),
        "recall": [
            float(v) for v in recall_score(
                y_true, pred, labels=list(range(len(CLASSES))),
                average=None, zero_division=0,
            )
        ],
    }


def summarize(rows):
    out = {}
    for method in sorted({r["method"] for r in rows}):
        subset = [r for r in rows if r["method"] == method]
        out[method] = {
            "accuracy": float(np.mean([r["accuracy"] for r in subset])),
            "accuracy_std": float(np.std([r["accuracy"] for r in subset])),
            "balanced_accuracy": float(
                np.mean([r["balanced_accuracy"] for r in subset])
            ),
            "balanced_std": float(
                np.std([r["balanced_accuracy"] for r in subset])
            ),
            "n": int(np.mean([r["n"] for r in subset])),
        }
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--shots", type=int, nargs="+", default=[3, 5, 10])
    parser.add_argument("--cal-fraction", type=float, default=0.3)
    parser.add_argument("--out-dir", default=REPORT_DIR)
    args = parser.parse_args()

    X, labels, sources, files = build_dataset(rebuild=False)
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder().fit(CLASSES)
    y = le.transform(labels)

    all_rows = []
    corpus_summary = {}
    t0 = time.time()

    for held_out in CORPORA:
        print("=" * 72)
        print(f"HELD-OUT FAMILY: {held_out}")
        print("=" * 72)
        train_idx = np.where(sources != held_out)[0]
        pool_idx = np.where(sources == held_out)[0]

        scaler, model = build_global_model(X[train_idx], y[train_idx])
        X_pool = scaler.transform(X[pool_idx])
        global_proba = model.predict_proba(X_pool)

        prior = PriorCorrector(list(range(len(CLASSES)))).fit(y[train_idx])
        prior_proba = prior.transform(global_proba)

        coral = CoralAdapter().fit(
            scaler.transform(X[train_idx]), X_pool
        )
        coral_proba = model.predict_proba(coral.transform(X_pool))
        X_pool_coral = coral.transform(X_pool)

        rows = []
        for seed in range(args.seeds):
            cal_idx, test_idx = split_recordings(
                files, y, pool_idx, args.cal_fraction, seed
            )
            pos = {idx: i for i, idx in enumerate(pool_idx)}
            test_pos = np.array([pos[i] for i in test_idx])
            y_true = y[test_idx]

            methods = {
                "global": global_proba[test_pos],
                "prior": prior_proba[test_pos],
                "coral": coral_proba[test_pos],
            }

            for shots in args.shots:
                shot_idx = sample_shots(cal_idx, y, shots, seed, files)
                if len(shot_idx) < len(CLASSES):
                    continue
                cal_pos = np.array([pos[i] for i in shot_idx])
                proto = PrototypeAdapter(
                    list(range(len(CLASSES)))
                ).fit(X_pool[cal_pos], y[shot_idx])
                proto_proba = proto.predict_proba(X_pool[test_pos])
                methods[f"proto-{shots}"] = proto_proba
                methods[f"blend-{shots}"] = blend_proba(
                    global_proba[test_pos], proto_proba, alpha=0.5
                )
                proto_coral = PrototypeAdapter(
                    list(range(len(CLASSES)))
                ).fit(X_pool_coral[cal_pos], y[shot_idx])
                methods[f"coral-proto-{shots}"] = proto_coral.predict_proba(
                    X_pool_coral[test_pos]
                )

            # Per-family model selection: use a held-out slice of the
            # family's own labelled clips to choose the adapter.
            fit_idx, val_idx = split_recordings(
                files, y, cal_idx, 0.7, seed + 1000
            )
            fit_pos = np.array([pos[i] for i in fit_idx])
            val_pos = np.array([pos[i] for i in val_idx])
            y_val = y[val_idx]
            val_scores = {
                "global": balanced_accuracy_score(
                    y_val, np.argmax(global_proba[val_pos], axis=1)
                ),
                "coral": balanced_accuracy_score(
                    y_val, np.argmax(coral_proba[val_pos], axis=1)
                ),
            }
            for shots in (10, 20):
                shot_fit = sample_shots(fit_idx, y, shots, seed, files)
                if len(shot_fit) < len(CLASSES):
                    continue
                shot_fit_pos = np.array([pos[i] for i in shot_fit])
                for prefix, space in (("", X_pool), ("coral-", X_pool_coral)):
                    candidate = PrototypeAdapter(
                        list(range(len(CLASSES)))
                    ).fit(space[shot_fit_pos], y[shot_fit])
                    val_scores[f"{prefix}proto-{shots}"] = balanced_accuracy_score(
                        y_val, np.argmax(candidate.predict_proba(
                            space[val_pos]), axis=1)
                    )

            best = max(val_scores, key=lambda name: (
                val_scores[name], val_scores.get(name, 0)
            ))
            if best in ("global", "coral"):
                methods["auto"] = methods[best]
            else:
                shots = int(best.rsplit("-", 1)[1])
                space = X_pool_coral if best.startswith("coral-") else X_pool
                shot_idx = sample_shots(cal_idx, y, shots, seed, files)
                cal_pos = np.array([pos[i] for i in shot_idx])
                candidate = PrototypeAdapter(
                    list(range(len(CLASSES)))
                ).fit(space[cal_pos], y[shot_idx])
                methods["auto"] = candidate.predict_proba(space[test_pos])
            auto_pick = best

            for method, proba in methods.items():
                result = score(y_true, proba)
                result.update({
                    "method": method, "seed": seed, "n": len(test_idx),
                    "corpus": held_out,
                })
                if method == "auto":
                    result["picked"] = auto_pick
                rows.append(result)

        summary = summarize(rows)
        corpus_summary[held_out] = summary
        all_rows.extend(rows)

        print(f"{'method':<12s} {'acc':>7s} {'bal':>7s}")
        for method, stats in summary.items():
            print(f"{method:<12s} {stats['accuracy']:>6.1%} "
                  f"{stats['balanced_accuracy']:>6.1%}")
        print()

    # Overall mean across corpora (weight by corpus equally)
    overall = {}
    methods = sorted({r["method"] for r in all_rows})
    for method in methods:
        per_corpus = [
            corpus_summary[c][method]
            for c in CORPORA if method in corpus_summary[c]
        ]
        overall[method] = {
            "accuracy": float(np.mean([s["accuracy"] for s in per_corpus])),
            "balanced_accuracy": float(
                np.mean([s["balanced_accuracy"] for s in per_corpus])
            ),
        }

    print("=" * 72)
    print("OVERALL (mean across corpora)")
    print("=" * 72)
    for method, stats in sorted(
        overall.items(), key=lambda kv: -kv[1]["balanced_accuracy"]
    ):
        print(f"{method:<12s} acc={stats['accuracy']:.1%} "
              f"bal={stats['balanced_accuracy']:.1%}")

    os.makedirs(args.out_dir, exist_ok=True)
    payload = {
        "classes": CLASSES,
        "seeds": args.seeds,
        "shots": args.shots,
        "cal_fraction": args.cal_fraction,
        "overall": overall,
        "per_corpus": corpus_summary,
        "rows": all_rows,
        "elapsed_seconds": round(time.time() - t0, 1),
    }
    json_path = os.path.join(args.out_dir, "adaptation.json")
    with open(json_path, "w") as f:
        json.dump(payload, f, indent=2)

    md_path = os.path.join(args.out_dir, "adaptation.md")
    with open(md_path, "w") as f:
        f.write("# Few-shot family calibration (leave-one-corpus-out)\n\n")
        f.write(
            "Each corpus is held out as a simulated new family. Calibration "
            f"uses {args.cal_fraction:.0%} of its recordings, split by "
            "recording; test uses the rest. Values are mean over "
            f"{args.seeds} seeds.\n\n"
        )
        f.write("| Method | Accuracy | Balanced |\n|---|---:|---:|\n")
        for method, stats in sorted(
            overall.items(), key=lambda kv: -kv[1]["balanced_accuracy"]
        ):
            f.write(
                f"| {method} | {stats['accuracy']:.1%} | "
                f"{stats['balanced_accuracy']:.1%} |\n"
            )
        f.write("\n## Per corpus\n\n")
        for corpus in CORPORA:
            f.write(f"### {corpus}\n\n")
            f.write("| Method | Accuracy | Balanced |\n|---|---:|---:|\n")
            for method, stats in sorted(
                corpus_summary[corpus].items(),
                key=lambda kv: -kv[1]["balanced_accuracy"],
            ):
                f.write(
                    f"| {method} | {stats['accuracy']:.1%} | "
                    f"{stats['balanced_accuracy']:.1%} |\n"
                )
            f.write("\n")
        f.write("## Auto picks\n\n")
        f.write("| Corpus | Picks (5 seeds) |\n|---|---|\n")
        for corpus in CORPORA:
            picks = Counter(
                r.get("picked") for r in all_rows
                if r["method"] == "auto" and r.get("corpus") == corpus
            )
            f.write(
                f"| {corpus} | "
                + ", ".join(f"{k}×{v}" for k, v in sorted(picks.items()))
                + " |\n"
            )

    print(f"\nSaved {md_path} and {json_path} "
          f"({payload['elapsed_seconds']}s)")


if __name__ == "__main__":
    main()
