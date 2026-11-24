"""End-to-end simulation with real audio.

Takes real .wav files from Donate-a-Cry, runs them through:
  audio → CryAudioClassifier → signals → LayeredJudge → final judgment

Compares Tachyone-only vs Tachyone+Jev layered approach.
"""
import glob
import os
import time
import random

from config import load_config
from audio_inference import CryAudioClassifier
from judge import LayeredJudge
from baby_profile import BabyProfile, CryEvent

DATA_DIR = os.path.expanduser(
    "~/.cache/huggingface/hub/"
    "datasets--Nooon--Donate_a_cry/snapshots/"
    "491487c4258287aa2be7cc179b7928768be7008d"
)
LABEL_MAP = {
    "belly_pain": "pain", "burping": "burp", "discomfort": "discomfort",
    "hungry": "hungry", "tired": "sleepy",
}


def collect_test_files(n_per_class=3):
    """Pick n files per class for testing."""
    test_files = []
    for class_name in os.listdir(DATA_DIR):
        class_dir = os.path.join(DATA_DIR, class_name)
        if not os.path.isdir(class_dir):
            continue
        wavs = sorted(glob.glob(os.path.join(class_dir, "*.wav")))
        random.seed(42)
        random.shuffle(wavs)
        for w in wavs[:n_per_class]:
            test_files.append((class_name, w))
    return test_files


def run():
    cfg = load_config()
    audio_clf = CryAudioClassifier()
    judge = LayeredJudge(cfg)
    profile = BabyProfile.load(
        os.path.expanduser("~/.config/cry-agent/profile.json")
    )

    test_files = collect_test_files(n_per_class=3)
    print(f"Test files: {len(test_files)}")
    print(f"Audio classifier classes: {audio_clf.classes}")
    print(f"Judge thresholds: high={cfg.tachyone_high} low={cfg.tachyone_low}")
    print()

    # Simulate different context for each file
    contexts = [
        {"feed_gap": 3.0, "sleep_gap": 4.0, "label": "realistic hungry"},
        {"feed_gap": 1.0, "sleep_gap": 4.0, "label": "realistic sleepy"},
        {"feed_gap": 1.5, "sleep_gap": 2.0, "label": "normal gap"},
        {"feed_gap": 0.5, "sleep_gap": 1.0, "label": "just fed"},
    ]

    results = []
    print(f"{'File':<30s} {'True':>10s} {'Audio':>10s} {'Conf':>5s}  "
          f"{'Judge':>10s} {'Conf':>5s} {'Model':>10s} {'Esc':>4s} {'Lat':>5s}")
    print("-" * 120)

    for i, (true_class, wav_file) in enumerate(test_files):
        true_mapped = LABEL_MAP.get(true_class, true_class)
        ctx = contexts[i % len(contexts)]

        # Set profile context
        profile.last_feed_time = time.time() - ctx["feed_gap"] * 3600
        profile.last_sleep_end = time.time() - ctx["sleep_gap"] * 3600

        # Tier 0: Audio classification
        t0 = time.perf_counter()
        audio_result = audio_clf.classify_file(wav_file)
        audio_latency = (time.perf_counter() - t0) * 1000

        # Build signals for judge
        cry_pattern = audio_clf.get_cry_pattern(audio_result["probabilities"])
        signals = {
            "cry_pattern": cry_pattern,
            "audio_confidence": audio_result["confidence"],
            "audio_top_cause": audio_result["cause"],
            "context_hint": ctx["label"],
        }

        # Tier 1+2: Layered judge
        judgment = judge.judge(signals, profile)

        is_correct = judgment.cause == true_mapped
        results.append({
            "true": true_mapped,
            "audio_pred": audio_result["cause"],
            "audio_conf": audio_result["confidence"],
            "judge_pred": judgment.cause,
            "judge_conf": judgment.confidence,
            "model": judgment.model,
            "correct": is_correct,
            "latency": audio_latency + judgment.latency_ms,
        })

        mark = "✓" if is_correct else "✗"
        fname = os.path.basename(wav_file)[:28]
        print(f" {mark} {fname:<28s} {true_mapped:>10s} "
              f"{audio_result['cause']:>10s} {audio_result['confidence']:>4.0%}  "
              f"{judgment.cause:>10s} {judgment.confidence:>4.0%} "
              f"{judgment.model:>10s} {'Y' if judgment.escalated else 'N':>4s} "
              f"{audio_latency + judgment.latency_ms:>4.0f}ms")

    # Summary
    print()
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)

    total = len(results)
    audio_correct = sum(1 for r in results if r["audio_pred"] == r["true"])
    judge_correct = sum(1 for r in results if r["correct"])
    escalated = sum(1 for r in results if r["model"] == "jev")
    avg_lat = sum(r["latency"] for r in results) / total

    print(f"Audio-only accuracy:   {audio_correct}/{total} ({audio_correct/total*100:.0f}%)")
    print(f"Layered judge accuracy: {judge_correct}/{total} ({judge_correct/total*100:.0f}%)")
    print(f"Escalated to Jev:       {escalated}/{total}")
    print(f"Avg total latency:      {avg_lat:.0f}ms (audio + judge)")

    # Per-class breakdown
    from collections import defaultdict
    per_class = defaultdict(lambda: {"correct": 0, "total": 0})
    for r in results:
        per_class[r["true"]]["total"] += 1
        if r["correct"]:
            per_class[r["true"]]["correct"] += 1

    print(f"\nPer-class recall:")
    for cls in sorted(per_class.keys()):
        pc = per_class[cls]
        print(f"  {cls:15s}: {pc['correct']}/{pc['total']} "
              f"({pc['correct']/pc['total']*100:.0f}%)")


if __name__ == "__main__":
    run()
