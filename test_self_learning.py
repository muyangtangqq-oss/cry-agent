"""Self-learning closed loop validation.

Simulates multiple days of usage with real audio from Donate-a-Cry:
  Day 1-3:   No personal data — pure audio classifier
  Day 4+:    Feedback starts adjusting signal weights
  Day 7+:    Personal cry library (embedding similarity) kicks in
  Day 14+:   Misjudgment correction notes fed to Jev

Measures: accuracy and confidence growth over time.
"""
import glob
import os
import time
import random
import json
import warnings
warnings.filterwarnings("ignore")

import numpy as np
from collections import Counter, defaultdict

from config import load_config
from audio_inference import CryAudioClassifier
from baby_profile import BabyProfile, CryEvent
from judge import LayeredJudge

DATA_DIR = os.path.expanduser(
    "~/.cache/huggingface/hub/"
    "datasets--Nooon--Donate_a_cry/snapshots/"
    "491487c4258287aa2be7cc179b7928768be7008d"
)
LABEL_MAP = {
    "belly_pain": "pain", "burping": "burp", "discomfort": "discomfort",
    "hungry": "hungry", "tired": "sleepy",
}


class FeedbackSimulator:
    """Simulates realistic parent feedback.

    - Correct judgment: parent confirms (90% of the time)
    - Wrong judgment: parent corrects (always)
    - Low confidence: parent checks and provides answer (80% of the time)
    """
    def __init__(self, seed=42):
        self.rng = random.Random(seed)

    def give_feedback(self, judgment, true_cause):
        if judgment.cause == true_cause:
            # Correct — parent confirms 90% of the time
            if self.rng.random() < 0.90:
                return "correct"
            else:
                return None  # parent didn't respond
        else:
            # Wrong — parent always corrects
            return true_cause


class PersonalCryLibrary:
    """Level 3: Personal cry embedding library.

    Stores audio embeddings per cause, retrieves similar past cries
    for comparison. Uses simple MFCC statistics as embedding.
    """
    def __init__(self):
        self.library = defaultdict(list)  # cause -> list of feature vectors

    def add(self, cause, features):
        if cause != "unknown":
            self.library[cause].append(features)

    def find_similar(self, features, top_k=3):
        """Find most similar past cries across all causes."""
        results = []
        for cause, vectors in self.library.items():
            for vec in vectors:
                dist = np.linalg.norm(features - vec)
                results.append((cause, dist))
        results.sort(key=lambda x: x[1])
        return results[:top_k]

    def size(self):
        return sum(len(v) for v in self.library.values())


class SelfLearningSimulator:
    def __init__(self):
        self.cfg = load_config()
        self.audio_clf = CryAudioClassifier()
        self.judge = LayeredJudge(self.cfg)
        self.profile = BabyProfile(
            age_days=42,
            feed_interval_mean_hours=2.5,
        )
        self.feedback_sim = FeedbackSimulator()
        self.cry_library = PersonalCryLibrary()
        self.profile_path = os.path.expanduser(
            "~/.config/cry-agent/profile.json"
        )

    def collect_files(self, n_per_class=20):
        """Collect all available files, grouped by class."""
        files = []
        for class_name in os.listdir(DATA_DIR):
            class_dir = os.path.join(DATA_DIR, class_name)
            if not os.path.isdir(class_dir):
                continue
            wavs = sorted(glob.glob(os.path.join(class_dir, "*.wav")))
            mapped = LABEL_MAP.get(class_name, class_name)
            for w in wavs:
                files.append((mapped, w))
        random.seed(42)
        random.shuffle(files)
        return files

    def extract_simple_embedding(self, y, sr):
        """Simple MFCC-based embedding for personal cry library."""
        import librosa
        if sr != 16000:
            y = librosa.resample(y, orig_sr=sr, target_sr=16000)
            sr = 16000
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        return mfcc.mean(axis=1)

    def simulate_day(self, day_num, daily_files):
        """Simulate one day of usage. Returns day stats."""
        correct = 0
        wrong = 0
        uncertain = 0
        total = len(daily_files)
        feedback_applied = 0

        # Simulate varying context throughout the day
        contexts = [
            (3.0, 4.0),  # morning: long gap since feed
            (1.0, 2.0),  # midday
            (2.5, 3.5),  # afternoon
            (0.5, 1.0),  # just fed
            (1.5, 4.5),  # evening
        ]

        for i, (true_cause, wav_file) in enumerate(daily_files):
            ctx = contexts[i % len(contexts)]
            self.profile.last_feed_time = time.time() - ctx[0] * 3600
            self.profile.last_sleep_end = time.time() - ctx[1] * 3600

            # Audio classification
            audio_result = self.audio_clf.classify_file(wav_file)

            # Extract embedding for personal library
            try:
                import librosa
                y, sr = librosa.load(wav_file, sr=None)
                embedding = self.extract_simple_embedding(y, sr)
            except:
                embedding = None

            # Build signals
            cry_pattern = self.audio_clf.get_cry_pattern(
                audio_result["probabilities"]
            )
            signals = {
                "cry_pattern": cry_pattern,
                "audio_confidence": audio_result["confidence"],
                "audio_top_cause": audio_result["cause"],
                "audio_probabilities": audio_result["probabilities"],
            }

            # Add personal library hint if available
            if embedding is not None and self.cry_library.size() > 0:
                similar = self.cry_library.find_similar(embedding, top_k=3)
                if similar:
                    similar_cause = similar[0][0]
                    similar_dist = similar[0][1]
                    # If very similar to a past cry, boost that cause
                    if similar_dist < 50:  # threshold
                        signals["personal_library_match"] = similar_cause
                        signals["personal_library_similarity"] = 1.0 / (1.0 + similar_dist / 10)

            # Judge
            judgment = self.judge.judge(signals, self.profile)

            # Record cry event
            event_idx = len(self.profile.cry_history)
            event = CryEvent(
                timestamp=time.time(),
                cause=judgment.cause,
                confidence=judgment.confidence,
                signals=signals,
            )
            self.profile.record_cry(event)

            # Get feedback
            feedback = self.feedback_sim.give_feedback(judgment, true_cause)

            if feedback == "correct":
                correct += 1
                # Add to personal library
                if embedding is not None:
                    self.cry_library.add(true_cause, embedding)
                # Positive feedback adjusts weights
                self.profile.record_feedback(event_idx, true_cause)
                feedback_applied += 1
            elif feedback is not None and feedback != "correct":
                wrong += 1
                # Wrong — add correct cause to library
                if embedding is not None:
                    self.cry_library.add(feedback, embedding)
                # Negative feedback adjusts weights
                self.profile.record_feedback(event_idx, feedback)
                feedback_applied += 1
            else:
                uncertain += 1
                # No feedback — still add to library if confident
                if embedding is not None and judgment.confidence > 0.7:
                    self.cry_library.add(judgment.cause, embedding)

        return {
            "day": day_num,
            "total": total,
            "correct": correct,
            "wrong": wrong,
            "uncertain": uncertain,
            "accuracy": correct / max(total - uncertain, 1),
            "feedback_applied": feedback_applied,
            "library_size": self.cry_library.size(),
            "signal_weights": dict(self.profile.signal_weights),
            "cry_history_size": len(self.profile.cry_history),
        }

    def run(self, days=14, events_per_day=15):
        files = self.collect_files()
        print(f"Total available files: {len(files)}")
        print(f"Simulating {days} days, {events_per_day} events/day")

        # Reset profile for clean experiment
        self.profile = BabyProfile(age_days=42, feed_interval_mean_hours=2.5)
        self.cry_library = PersonalCryLibrary()

        # Need to start Tachyone server
        print(f"\nStarting Tachyone server...")
        os.system("python3.12 tachyone_server.py 8901 &")
        time.sleep(1)

        print(f"\n{'Day':>4s} {'Events':>7s} {'Correct':>8s} {'Wrong':>6s} "
              f"{'Uncert':>7s} {'Accuracy':>9s} {'Feedback':>9s} "
              f"{'Library':>8s} {'Weights':>40s}")
        print("-" * 110)

        all_stats = []
        file_idx = 0

        for day in range(1, days + 1):
            # Select files for this day
            day_files = []
            for j in range(events_per_day):
                if file_idx >= len(files):
                    file_idx = 0  # cycle through files
                day_files.append(files[file_idx])
                file_idx += 1

            stats = self.simulate_day(day, day_files)
            all_stats.append(stats)

            weights_str = ", ".join(
                f"{k[:6]}={v:.2f}" for k, v in stats["signal_weights"].items()
            )
            print(f"{day:4d} {stats['total']:7d} {stats['correct']:8d} "
                  f"{stats['wrong']:6d} {stats['uncertain']:7d} "
                  f"{stats['accuracy']:8.0%} {stats['feedback_applied']:9d} "
                  f"{stats['library_size']:8d} {weights_str}")

        # Save profile
        self.profile.save(self.profile_path)

        # Cleanup
        os.system("kill %1 2>/dev/null")

        # Summary
        print(f"\n{'='*70}")
        print("SELF-LEARNING GROWTH CURVE")
        print(f"{'='*70}")
        print(f"{'Day':>4s} {'Accuracy':>9s} {'Library':>8s} {'Top Weight':>12s}")
        print("-" * 40)
        for s in all_stats:
            top_weight = max(s["signal_weights"].values())
            print(f"{s['day']:4d} {s['accuracy']:8.0%} "
                  f"{s['library_size']:8d} {top_weight:11.2f}")

        # Compare early vs late
        early = all_stats[:3]
        late = all_stats[-3:]
        early_acc = sum(s["accuracy"] for s in early) / len(early)
        late_acc = sum(s["accuracy"] for s in late) / len(late)
        print(f"\nEarly accuracy (Day 1-3):   {early_acc:.0%}")
        print(f"Late accuracy  (Day {days-2}-{days}):  {late_acc:.0%}")
        print(f"Improvement: {late_acc - early_acc:+.0%}")

        # Weight evolution
        print(f"\nSignal weight evolution:")
        print(f"  Initial: {dict(all_stats[0]['signal_weights'])}")
        print(f"  Final:   {dict(all_stats[-1]['signal_weights'])}")

        return all_stats


if __name__ == "__main__":
    sim = SelfLearningSimulator()
    sim.run(days=14, events_per_day=15)
