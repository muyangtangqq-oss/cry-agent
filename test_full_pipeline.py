"""Full end-to-end pipeline test.

Combines:
  1. Real audio file → audio classification
  2. Vision annotation → posture detection
  3. Context → baby profile
  4. Env → simulated sensors
  5. Multimodal fusion → layered judge

Tests 6 scenarios with real audio + simulated vision/env.
"""
import glob
import os
import time
import tempfile

from config import load_config
from pipeline import CryAgentPipeline
from baby_profile import BabyProfile
from vision_detector import VisionDetector, create_annotation_file

DONATE_DIR = os.path.expanduser(
    "~/.cache/huggingface/hub/datasets--Nooon--Donate_a_cry/snapshots/"
    "491487c4258287aa2be7cc179b7928768be7008d"
)
LABEL_MAP = {
    "belly_pain": "pain", "burping": "burp", "discomfort": "discomfort",
    "hungry": "hungry", "tired": "sleepy",
}

# Pick one sample per class
def get_test_files():
    files = {}
    for cls_name in os.listdir(DONATE_DIR):
        cls_dir = os.path.join(DONATE_DIR, cls_name)
        if not os.path.isdir(cls_dir):
            continue
        mapped = LABEL_MAP.get(cls_name, cls_name)
        wavs = sorted(glob.glob(os.path.join(cls_dir, "*.wav")))
        if wavs:
            files[mapped] = wavs[0]
    return files


def run():
    cfg = load_config()
    pipeline = CryAgentPipeline(cfg)
    vision = VisionDetector()

    test_files = get_test_files()
    print("=" * 70)
    print("FULL END-TO-END PIPELINE TEST")
    print("Audio (real) + Vision (annotated) + Context + Env → Fusion → Judge")
    print("=" * 70)
    print(f"Test files: {len(test_files)} classes")
    print()

    # Define scenarios: vision signals that may or may not agree with audio
    scenarios = [
        {
            "audio_class": "hungry",
            "postures": ["rooting", "sucking_fingers"],
            "face": None, "temp": 23, "noise": 40,
            "feed_gap": 3.0, "sleep_gap": 2.0,
            "name": "hungry: audio+vision agree",
        },
        {
            "audio_class": "sleepy",
            "postures": ["rubbing_eyes", "yawning"],
            "face": None, "temp": 22, "noise": 35,
            "feed_gap": 1.0, "sleep_gap": 4.5,
            "name": "sleepy: audio+vision+context agree",
        },
        {
            "audio_class": "pain",
            "postures": ["arching_back", "clenched_fists"],
            "face": None, "temp": 24, "noise": 50,
            "feed_gap": 0.5, "sleep_gap": 1.0,
            "name": "pain: critical posture",
        },
        {
            "audio_class": "discomfort",
            "postures": ["fetal_position", "squirming"],
            "face": None, "temp": 18, "noise": 30,
            "feed_gap": 1.5, "sleep_gap": 2.0,
            "name": "discomfort: cold env overrides",
        },
        {
            "audio_class": "burp",
            "postures": [],
            "face": None, "temp": 23, "noise": 35,
            "feed_gap": 0.5, "sleep_gap": 1.0,
            "name": "burp: audio only, context supports",
        },
        {
            "audio_class": "hungry",
            "postures": ["reaching_up", "looking_around"],
            "face": None, "temp": 23, "noise": 35,
            "feed_gap": 1.0, "sleep_gap": 2.0,
            "name": "lonely: vision overrides hungry audio",
        },
    ]

    correct = 0
    total = 0

    print(f"{'Scenario':<40s} {'Audio':>10s} {'Fused':>10s} "
          f"{'Final':>10s} {'Model':>8s} {'Conf':>5s} {'Lat':>5s}")
    print("-" * 100)

    for sc in scenarios:
        wav_file = test_files.get(sc["audio_class"])
        if not wav_file:
            print(f"  SKIP: no audio for {sc['audio_class']}")
            continue

        profile = BabyProfile(age_days=42)
        profile.last_feed_time = time.time() - sc["feed_gap"] * 3600
        profile.last_sleep_end = time.time() - sc["sleep_gap"] * 3600

        # Create vision annotation
        tmpdir = tempfile.mkdtemp()
        img_path = os.path.join(tmpdir, "frame.png")
        with open(img_path, "wb") as f:
            f.write(b'\x89PNG\r\n\x1a\n')
        if sc["postures"] or sc["face"]:
            create_annotation_file(img_path, sc["postures"], sc["face"])
        vision_result = vision.detect_from_image(img_path)

        # Run pipeline
        result = pipeline.run(
            audio_path=wav_file,
            postures=vision_result["postures"],
            face_state=vision_result["face_state"],
            room_temp=sc["temp"],
            ambient_noise_db=sc["noise"],
            profile=profile,
        )

        # Determine expected
        if sc["name"].startswith("lonely"):
            expected = "lonely"
        elif sc["name"].startswith("discomfort") and sc["temp"] < 20:
            expected = "temperature"
        else:
            expected = sc["audio_class"]

        is_correct = result.cause == expected
        if is_correct:
            correct += 1
        total += 1

        mark = "✓" if is_correct else "✗"
        print(f" {mark} {sc['name']:<37s} {result.audio_cause:>10s} "
              f"{result.fused_cause:>10s} {result.cause:>10s} "
              f"{result.model:>8s} {result.confidence:>4.0%} "
              f"{result.latency_ms:>4.0f}ms")

    print(f"\n{'='*70}")
    print(f"Pipeline accuracy: {correct}/{total} ({correct/max(total,1)*100:.0f}%)")
    print(f"{'='*70}")


if __name__ == "__main__":
    run()
