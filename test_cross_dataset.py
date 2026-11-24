"""Cross-dataset generalization test.

Train on Donate-a-Cry, test on InfantCry-DBL — measures how well
the classifier generalizes to unseen data from a different source.

Also runs the self-learning loop on the new dataset to show
accuracy improvement over time with feedback.
"""
import glob
import os
import time
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import librosa
from collections import Counter, defaultdict
import joblib

from config import load_config
from audio_inference import CryAudioClassifier
from baby_profile import BabyProfile, CryEvent
from judge import LayeredJudge

DBL_DIR = "/Users/kl/Documents/Codex/2026-11-23/you-2/work/cry-agent/data/infantcry-dbl-v2/dunstan  337"

# Dunstan baby language -> our labels
DUNSTAN_MAP = {
    "neh": "hungry",      # "Neh" = hungry
    "owh": "sleepy",      # "Owh" = sleepy
    "heh": "discomfort",  # "Heh" = discomfort
    "eairh": "pain",      # "Eairh" = lower gas/pain
    "eh": "burp",         # "Eh" = needs burping
}


def load_dbl_files():
    """Load all InfantCry-DBL files with true labels."""
    files = []
    for dunstan_label, mapped in DUNSTAN_MAP.items():
        class_dir = os.path.join(DBL_DIR, dunstan_label)
        wavs = sorted(glob.glob(os.path.join(class_dir, "*.wav")))
        for w in wavs:
            files.append((mapped, w))
    return files


def test_cross_dataset():
    """Test classifier trained on Donate-a-Cry against InfantCry-DBL."""
    print("=" * 70)
    print("CROSS-DATASET GENERALIZATION TEST")
    print("Training: Donate-a-Cry (457 clips, 5 classes)")
    print("Testing:  InfantCry-DBL v2 (337 clips, 5 Dunstan classes)")
    print("=" * 70)

    audio_clf = CryAudioClassifier()
    files = load_dbl_files()
    print(f"\nTest files: {len(files)}")
    print(f"Label distribution: {Counter(l for l, _ in files)}")
    print()

    # Direct classification test
    print(f"{'File':<35s} {'True':>12s} {'Pred':>12s} {'Conf':>6s}  Top-3")
    print("-" * 100)

    correct = 0
    total = 0
    per_class = defaultdict(lambda: {"correct": 0, "total": 0})
    low_conf_count = 0

    for true_label, wav_file in files:
        try:
            result = audio_clf.classify_file(wav_file)
        except Exception as e:
            print(f"  SKIP {os.path.basename(wav_file)}: {e}")
            continue

        pred = result["cause"]
        conf = result["confidence"]
        is_correct = pred == true_label
        total += 1
        if is_correct:
            correct += 1
        per_class[true_label]["total"] += 1
        if is_correct:
            per_class[true_label]["correct"] += 1
        if conf < 0.80:
            low_conf_count += 1

        mark = "✓" if is_correct else "✗"
        fname = os.path.basename(wav_file)[:33]
        top3 = audio_clf.get_cry_pattern(result["probabilities"])
        print(f" {mark} {fname:<33s} {true_label:>12s} {pred:>12s} "
              f"{conf:>5.0%}  {top3}")

    print(f"\n{'='*70}")
    print("CROSS-DATASET RESULTS")
    print(f"{'='*70}")
    print(f"Overall accuracy: {correct}/{total} ({correct/total*100:.1f}%)")
    print(f"Low confidence (<80%): {low_conf_count}/{total} ({low_conf_count/total*100:.1f}%)")
    print(f"\nPer-class recall:")
    for cls in sorted(per_class.keys()):
        pc = per_class[cls]
        print(f"  {cls:15s}: {pc['correct']}/{pc['total']} "
              f"({pc['correct']/pc['total']*100:.0f}%)")

    return correct, total, files


def test_self_learning_on_dbl(files):
    """Run self-learning loop on InfantCry-DBL data.

    Simulates 7 days of usage with parent feedback on the new dataset.
    """
    print(f"\n{'='*70}")
    print("SELF-LEARNING ON NEW DATASET (7-day simulation)")
    print(f"{'='*70}")

    cfg = load_config()
    audio_clf = CryAudioClassifier()
    judge = LayeredJudge(cfg)
    profile = BabyProfile(age_days=42)

    # Personal cry library (simple MFCC embedding)
    library = defaultdict(list)

    import random
    rng = random.Random(42)
    shuffled = list(files)
    rng.shuffle(shuffled)

    days = 7
    per_day = len(shuffled) // days
    print(f"Files: {len(shuffled)}, {per_day}/day for {days} days\n")

    print(f"{'Day':>4s} {'Events':>7s} {'Correct':>8s} {'Wrong':>6s} "
          f"{'Uncert':>7s} {'Accuracy':>9s} {'Library':>8s} "
          f"{'cry_w':>6s} {'post_w':>7s} {'ctx_w':>6s}")
    print("-" * 85)

    all_stats = []
    contexts = [(3.0, 4.0), (1.0, 2.0), (2.5, 3.5), (0.5, 1.0), (1.5, 4.5)]

    for day in range(1, days + 1):
        day_files = shuffled[(day-1)*per_day : day*per_day]
        correct = wrong = uncertain = 0
        feedback_count = 0

        for i, (true_cause, wav_file) in enumerate(day_files):
            ctx = contexts[i % len(contexts)]
            profile.last_feed_time = time.time() - ctx[0] * 3600
            profile.last_sleep_end = time.time() - ctx[1] * 3600

            try:
                audio_result = audio_clf.classify_file(wav_file)
            except:
                continue

            # Extract embedding
            try:
                y, sr = librosa.load(wav_file, sr=None)
                if sr != 16000:
                    y = librosa.resample(y, orig_sr=sr, target_sr=16000)
                mfcc = librosa.feature.mfcc(y=y, sr=16000, n_mfcc=13)
                embedding = mfcc.mean(axis=1)
            except:
                embedding = None

            signals = {
                "cry_pattern": audio_clf.get_cry_pattern(audio_result["probabilities"]),
                "audio_confidence": audio_result["confidence"],
                "audio_top_cause": audio_result["cause"],
                "audio_probabilities": audio_result["probabilities"],
            }

            # Personal library hint
            if embedding is not None and sum(len(v) for v in library.values()) > 0:
                similar = []
                for cause, vectors in library.items():
                    for vec in vectors:
                        dist = np.linalg.norm(embedding - vec)
                        similar.append((cause, dist))
                similar.sort(key=lambda x: x[1])
                if similar and similar[0][1] < 50:
                    signals["personal_library_match"] = similar[0][0]
                    signals["personal_library_similarity"] = 1.0 / (1.0 + similar[0][1] / 10)

            judgment = judge.judge(signals, profile)

            event_idx = len(profile.cry_history)
            event = CryEvent(
                timestamp=time.time(),
                cause=judgment.cause,
                confidence=judgment.confidence,
                signals=signals,
            )
            profile.record_cry(event)

            # Simulate feedback
            if judgment.cause == true_cause:
                if rng.random() < 0.85:
                    correct += 1
                    if embedding is not None:
                        library[true_cause].append(embedding)
                    profile.record_feedback(event_idx, true_cause)
                    feedback_count += 1
                else:
                    uncertain += 1
            else:
                wrong += 1
                if embedding is not None:
                    library[true_cause].append(embedding)
                profile.record_feedback(event_idx, true_cause)
                feedback_count += 1

        total = correct + wrong + uncertain
        acc = correct / max(total - uncertain, 1)
        lib_size = sum(len(v) for v in library.values())
        w = profile.signal_weights

        all_stats.append({
            "day": day, "accuracy": acc, "library": lib_size,
            "correct": correct, "wrong": wrong, "uncertain": uncertain,
        })

        print(f"{day:4d} {total:7d} {correct:8d} {wrong:6d} {uncertain:7d} "
              f"{acc:8.0%} {lib_size:8d} "
              f"{w['cry_pattern']:.2f}  {w['posture']:.2f}   {w['context']:.2f}")

    # Growth curve
    print(f"\n{'='*70}")
    print("GROWTH CURVE")
    print(f"{'='*70}")
    early = all_stats[:2]
    late = all_stats[-2:]
    early_acc = sum(s["accuracy"] for s in early) / len(early)
    late_acc = sum(s["accuracy"] for s in late) / len(late)
    print(f"Early (Day 1-2):  {early_acc:.0%}")
    print(f"Late  (Day {days-1}-{days}):  {late_acc:.0%}")
    print(f"Improvement: {late_acc - early_acc:+.0%}")


if __name__ == "__main__":
    correct, total, files = test_cross_dataset()
    test_self_learning_on_dbl(files)
