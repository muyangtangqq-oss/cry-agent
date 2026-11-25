"""Semantic + acoustic audit of the Dunstan label mapping.

The classifier reuses InfantCry-DBL, whose five labels are Dunstan reflex
sounds, not verified causes. This script answers three questions:

  1. What does the corpus itself say each label means, and how well does
     our 8-class cause taxonomy match it?
  2. Are the five sounds acoustically separable inside the corpus?
  3. Does `eairh -> pain` hold up, or is `eairh` closer to discomfort?

It also uses both tiers: the previous training/eval used only the 337
Dunstan-Core clips, leaving 1,214 Tier-2 clips unused.

Usage:
  python audit_dunstan.py
  python audit_dunstan.py --rebuild-cache
"""
import argparse
import csv
import json
import os
import time
import warnings
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor

warnings.filterwarnings("ignore")

import numpy as np

from train_combined_v2 import SR, extract_features, segs

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                    "data", "infantcry-dbl-v2")
META = os.path.join(ROOT, "metadata.csv")
CACHE = os.path.expanduser("~/.cache/cry-agent/dbl_all_features_v1.npz")
LOCO_CACHE = os.path.expanduser("~/.cache/cry-agent/loco_features_v1.npz")
REPORT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "reports")

DUNSTAN = ["eairh", "eh", "heh", "neh", "owh"]

# Our previous mapping. `eairh` is the questionable one.
MAPPING_A = {
    "neh": "hungry", "owh": "sleepy", "heh": "discomfort",
    "eairh": "pain", "eh": "burp",
}
# Semantics-preserving alternative: lower-wind discomfort, not sharp pain.
MAPPING_B = dict(MAPPING_A, eairh="discomfort")

DEFINITIONS = {
    "neh": ("hunger", "sucking reflex against the soft palate"),
    "owh": ("sleepiness", "yawn-like rounding"),
    "heh": ("physical discomfort", "skin-related stimulus"),
    "eairh": ("lower-wind / abdominal discomfort", "straining, lower gas"),
    "eh": ("upper-wind / need to burp", "burp reflex"),
}


def _one_file(args):
    rel, label, tier, source = args
    import librosa

    path = os.path.join(ROOT, rel)
    if not os.path.exists(path) and rel.startswith("1214 COLLCTED DATA/"):
        # metadata.csv has a typo; the archive folder is spelled correctly.
        path = os.path.join(
            ROOT, rel.replace("1214 COLLCTED DATA/", "1214 Collected Data/", 1)
        )
    try:
        y, sr = librosa.load(path, sr=SR, mono=True)
    except Exception:
        return []
    rows = []
    for seg in segs(y, sr):
        feat = extract_features(seg, sr)
        if feat is not None:
            rows.append((feat, label, tier, source, rel))
    return rows


def build_cache(rebuild=False, jobs=0):
    if os.path.exists(CACHE) and not rebuild:
        data = np.load(CACHE, allow_pickle=False)
        return (data["X"], data["label"], data["tier"],
                data["source"], data["files"])

    with open(META) as f:
        meta = list(csv.DictReader(f))
    payload = [
        (row["relative_path"], row["class"], row["tier_name"], row["source"])
        for row in meta
    ]
    jobs = jobs or max(1, (os.cpu_count() or 2) - 1)
    print(f"  extracting features from {len(payload)} files ({jobs} workers)")
    rows = []
    with ProcessPoolExecutor(max_workers=jobs) as pool:
        for chunk in pool.map(_one_file, payload, chunksize=8):
            rows.extend(chunk)
    X = np.asarray([r[0] for r in rows], dtype=np.float32)
    labels = np.asarray([r[1] for r in rows])
    tiers = np.asarray([r[2] for r in rows])
    sources = np.asarray([r[3] for r in rows])
    files = np.asarray([r[4] for r in rows])
    os.makedirs(os.path.dirname(CACHE), exist_ok=True)
    np.savez_compressed(
        CACHE, X=X, label=labels, tier=tiers, source=sources, files=files
    )
    print(f"  cached {X.shape} -> {CACHE}")
    return X, labels, tiers, sources, files


def make_model(labels):
    from test_loco import make_eval_pipeline
    return make_eval_pipeline(labels)


def evaluate(y_true, pred, classes):
    from sklearn.metrics import (
        accuracy_score, balanced_accuracy_score, confusion_matrix, recall_score,
    )
    ids = list(range(len(classes)))
    return {
        "n": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, pred)),
        "recall": [float(v) for v in recall_score(
            y_true, pred, labels=ids, average=None, zero_division=0)],
        "confusion": confusion_matrix(y_true, pred, labels=ids).tolist(),
    }


def five_class_experiments(X, labels, tiers, files):
    from sklearn.model_selection import StratifiedGroupKFold
    from sklearn.preprocessing import LabelEncoder

    le = LabelEncoder().fit(DUNSTAN)
    y = le.transform(labels)
    results = {}

    print("\n[2/5] Within-corpus CV (Dunstan sounds, 5 classes)")
    for tier in ("Dunstan-Core", "InfantCry-1214"):
        idx = np.where(tiers == tier)[0]
        if len(idx) < 50:
            continue
        splitter = StratifiedGroupKFold(
            n_splits=5, shuffle=True, random_state=42
        )
        pred = np.zeros(len(idx), dtype=int)
        for train_pos, test_pos in splitter.split(
                X[idx], y[idx], groups=files[idx]):
            model = make_model(y[idx][train_pos])
            model.fit(X[idx][train_pos], y[idx][train_pos])
            pred[test_pos] = model.predict(X[idx][test_pos])
        result = evaluate(y[idx], pred, DUNSTAN)
        result["majority"] = float(
            np.mean(y[idx] == np.bincount(y[idx]).argmax())
        )
        results[f"cv_{tier}"] = result
        print(f"  {tier:<16s} n={result['n']:<5d} "
              f"acc={result['accuracy']:.1%} "
              f"bal={result['balanced_accuracy']:.1%} "
              f"majority={result['majority']:.1%}")

    print("\n[3/5] Cross-tier (same taxonomy, different recording setup)")
    for train_tier, test_tier in (
        ("Dunstan-Core", "InfantCry-1214"),
        ("InfantCry-1214", "Dunstan-Core"),
    ):
        tr = np.where(tiers == train_tier)[0]
        te = np.where(tiers == test_tier)[0]
        model = make_model(y[tr])
        model.fit(X[tr], y[tr])
        pred = model.predict(X[te])
        result = evaluate(y[te], pred, DUNSTAN)
        result["majority"] = float(
            np.mean(y[te] == np.bincount(y[tr]).argmax())
        )
        key = f"{train_tier}->{test_tier}"
        results[key] = result
        print(f"  {train_tier:<16s} -> {test_tier:<16s} "
              f"acc={result['accuracy']:.1%} "
              f"bal={result['balanced_accuracy']:.1%} "
              f"majority={result['majority']:.1%}")
    return results


def cross_corpus_mapping(X, labels, tiers, files):
    """Train on Donate+Dynamic, test on DBL under mapping A vs B."""
    from sklearn.preprocessing import LabelEncoder

    if not os.path.exists(LOCO_CACHE):
        print("  skipped: LOCO cache missing")
        return {}
    data = np.load(LOCO_CACHE, allow_pickle=False)
    lx, ll, lsrc = data["X"], data["y"], data["source"]
    train = np.where(lsrc != "dbl")[0]
    le = LabelEncoder().fit(["burp", "discomfort", "hungry", "pain", "sleepy"])
    y_train = le.transform(ll[train])
    model = make_model(y_train)
    model.fit(lx[train], y_train)

    print("\n[4/5] Cross-corpus: Donate+Dynamic -> InfantCry-DBL")
    results = {}
    for name, mapping in (("A_eairh_pain", MAPPING_A),
                          ("B_eairh_discomfort", MAPPING_B)):
        mapped = np.array([mapping[v] for v in labels])
        y_mapped = le.transform(mapped)
        for tier in ("Dunstan-Core", "InfantCry-1214"):
            idx = np.where(tiers == tier)[0]
            y_true = y_mapped[idx]
            pred = model.predict(X[idx])
            result = evaluate(y_true, pred, list(le.classes_))
            result["majority"] = float(
                np.mean(y_true == np.bincount(y_train).argmax())
            )
            results[f"{name}/{tier}"] = result
            print(f"  {name:<20s} {tier:<16s} "
                  f"acc={result['accuracy']:.1%} "
                  f"bal={result['balanced_accuracy']:.1%}")

        # Does adding the unused Tier 2 to training help Tier 1?
        t2 = np.where(tiers == "InfantCry-1214")[0]
        t1 = np.where(tiers == "Dunstan-Core")[0]
        model2 = make_model(np.concatenate([y_train, y_mapped[t2]]))
        model2.fit(
            np.vstack([lx[train], X[t2]]),
            np.concatenate([y_train, y_mapped[t2]]),
        )
        result = evaluate(y_mapped[t1], model2.predict(X[t1]),
                          list(le.classes_))
        result["majority"] = float(
            np.mean(y_mapped[t1] == np.bincount(y_train).argmax())
        )
        key = f"{name}_plus_tier2/Dunstan-Core"
        results[key] = result
        print(f"  {name}_plus_tier2      Dunstan-Core     "
              f"acc={result['accuracy']:.1%} "
              f"bal={result['balanced_accuracy']:.1%}")

    # What does the model do with eairh specifically?
    eairh = np.where(labels == "eairh")[0]
    pred = model.predict(X[eairh])
    results["eairh_predictions"] = dict(
        Counter(le.classes_[p] for p in pred)
    )
    print("  eairh predicted as:", results["eairh_predictions"])
    return results


def acoustic_profile(X, labels):
    rows = {}
    for cls in DUNSTAN:
        mask = labels == cls
        rows[cls] = {
            "segments": int(mask.sum()),
            "centroid_mean": round(float(X[mask, 80].mean()), 1),
            "rms_mean": round(float(X[mask, 90].mean()), 4),
            "f0_mean": round(float(np.nanmean(X[mask, 92])), 1),
        }
    return rows


def metadata_audit():
    with open(META) as f:
        rows = list(csv.DictReader(f))
    notes = [
        {
            "file": r["relative_path"],
            "class": r["class"],
            "note": r["notes"],
        }
        for r in rows if r["notes"].strip()
    ]
    missing = [
        r["relative_path"] for r in rows
        if not os.path.exists(os.path.join(ROOT, r["relative_path"]))
    ]
    return {
        "clips": len(rows),
        "tiers": dict(Counter(r["tier_name"] for r in rows)),
        "classes": dict(Counter(r["class"] for r in rows)),
        "by_tier_class": {
            tier: dict(Counter(r["class"] for r in rows
                               if r["tier_name"] == tier))
            for tier in ("Dunstan-Core", "InfantCry-1214")
        },
        "sources": dict(Counter(r["source"] for r in rows)),
        "source_confirmed": dict(
            Counter(r["source_confirmed"] for r in rows)
        ),
        "duration_minutes": round(
            sum(float(r["duration_s"]) for r in rows) / 60, 1
        ),
        "label_notes": notes,
        "missing_paths": len(missing),
        "missing_path_examples": missing[:3],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rebuild-cache", action="store_true")
    parser.add_argument("--jobs", type=int, default=0)
    parser.add_argument("--out-dir", default=REPORT_DIR)
    args = parser.parse_args()
    t0 = time.time()

    print("=" * 72)
    print("DUNSTAN LABEL-SEMANTICS AUDIT")
    print("=" * 72)
    print("[1/5] Metadata")
    meta = metadata_audit()
    print(f"  clips={meta['clips']} minutes={meta['duration_minutes']}")
    print(f"  tiers={meta['tiers']}")
    print(f"  classes={meta['classes']}")

    print("\n[1b/5] Feature cache")
    X, labels, tiers, sources, files = build_cache(
        args.rebuild_cache, args.jobs
    )
    print(f"  X={X.shape} segments by tier={dict(Counter(tiers))}")

    five = five_class_experiments(X, labels, tiers, files)
    mapped = cross_corpus_mapping(X, labels, tiers, files)
    profile = acoustic_profile(X, labels)

    payload = {
        "metadata": meta,
        "definitions": DEFINITIONS,
        "mapping_A": MAPPING_A,
        "mapping_B": MAPPING_B,
        "five_class": five,
        "cross_corpus": mapped,
        "acoustic_profile": profile,
        "elapsed_seconds": round(time.time() - t0, 1),
    }
    os.makedirs(args.out_dir, exist_ok=True)
    json_path = os.path.join(args.out_dir, "dunstan_audit.json")
    with open(json_path, "w") as f:
        json.dump(payload, f, indent=2)

    md_path = os.path.join(args.out_dir, "dunstan_audit.md")
    with open(md_path, "w") as f:
        f.write("# Dunstan label-semantics audit\n\n")
        f.write(
            "Source definitions are from the local "
            "`data/infantcry-dbl-v2/README.md` and `metadata.csv`.\n\n"
        )
        f.write("## 1. What the labels actually mean\n\n")
        f.write("| DBL | Corpus definition | Our mapping | Fit |\n")
        f.write("|---|---|---|---|\n")
        for cls in DUNSTAN:
            meaning, detail = DEFINITIONS[cls]
            ours = MAPPING_A[cls]
            fit = "direct" if cls != "eairh" else "**over-strong**"
            f.write(
                f"| {cls} | {meaning} ({detail}) | {ours} | {fit} |\n"
            )
        f.write(
            "\n`eairh` is defined as lower-wind/abdominal discomfort. "
            "Mapping it to `pain` conflates gas discomfort with sharp pain; "
            "`discomfort` is the closer label. `heh` is skin-related "
            "discomfort and `eh` is upper-wind/burp, both direct matches.\n\n"
        )
        f.write("## 2. Corpus composition\n\n")
        f.write(f"- Clips: {meta['clips']} ({meta['duration_minutes']} min)\n")
        for tier, count in meta["tiers"].items():
            f.write(f"- {tier}: {count} clips, "
                    f"{meta['by_tier_class'][tier]}\n")
        f.write(f"- Sources: {meta['sources']}\n")
        f.write(f"- source_confirmed: {meta['source_confirmed']}\n")
        if meta["label_notes"]:
            f.write("\nLabel inconsistencies recorded in metadata:\n\n")
            for note in meta["label_notes"]:
                f.write(f"- `{note['file']}`: {note['note']}\n")
        if meta["missing_paths"]:
            f.write(
                f"\n`metadata.csv` paths do not resolve for "
                f"{meta['missing_paths']} rows (example: "
                f"`{meta['missing_path_examples'][0]}`). The archive folder "
                "is `1214 Collected Data` while the CSV says "
                "`1214 COLLCTED DATA`; the loader rewrites the prefix.\n"
            )
        f.write(
            "\nTier 1 is the 337 studio clips used by the previous "
            "training/eval; Tier 2's 1,214 heterogeneous clips were unused. "
            "All Tier-2 `source_confirmed` values are FALSE and 746 of them "
            "have source `unknown`.\n\n"
        )
        f.write("## 3. Are the five sounds separable?\n\n")
        f.write("| Experiment | n | Acc | Balanced | Majority |\n")
        f.write("|---|---:|---:|---:|---:|\n")
        for key, r in five.items():
            f.write(f"| {key} | {r['n']} | {r['accuracy']:.1%} | "
                    f"{r['balanced_accuracy']:.1%} | {r['majority']:.1%} |\n")
        f.write(
            "\nWithin-corpus separation is the ceiling for any mapping: if "
            "the sounds are not separable even inside one tier, the label "
            "mapping is not the bottleneck. Cross-tier rows show how much "
            "recording setup alone changes the result.\n\n"
        )
        f.write("## 4. Mapping A vs B, trained on Donate+Dynamic\n\n")
        f.write("| Mapping / test tier | n | Acc | Balanced |\n")
        f.write("|---|---:|---:|---:|\n")
        for key, r in mapped.items():
            if not isinstance(r, dict) or "accuracy" not in r:
                continue
            f.write(f"| {key} | {r['n']} | {r['accuracy']:.1%} | "
                    f"{r['balanced_accuracy']:.1%} |\n")
        if "eairh_predictions" in mapped:
            f.write(f"\n`eairh` predictions: {mapped['eairh_predictions']}\n")
        f.write("\n## 5. Acoustic profile by Dunstan class\n\n")
        f.write("| Class | Segments | Centroid | RMS | F0 mean |\n")
        f.write("|---|---:|---:|---:|---:|\n")
        for cls, row in profile.items():
            f.write(f"| {cls} | {row['segments']} | {row['centroid_mean']} | "
                    f"{row['rms_mean']} | {row['f0_mean']} |\n")
        f.write("\n## Conclusion\n\n")
        f.write(
            "1. The mapping is mostly direct, but `eairh -> pain` is stronger "
            "than the corpus's own definition (lower-wind / abdominal "
            "discomfort). It is retained in production because DBL's `eairh` "
            "segments are 36% of all pain training data and re-mapping to "
            "`discomfort` costs pain recall; see "
            "`reports/eairh_mapping.md` for the A/B benchmark.\n"
            "2. The corpus is five *reflex sounds*, not five verified causes. "
            "Even a perfect DBL classifier is not a cause diagnosis.\n"
            "3. Only Tier 1 was used before. Tier 2 adds 1,214 clips but is "
            "unconfirmed and heterogeneous; it is useful for domain-shift "
            "stress tests, not as clean ground truth.\n"
            "4. Any cross-corpus number should be reported with the mapping "
            "and tier, because both change the label and the domain.\n"
        )

    print(f"\nSaved {md_path} and {json_path} "
          f"({payload['elapsed_seconds']}s)")


if __name__ == "__main__":
    main()
