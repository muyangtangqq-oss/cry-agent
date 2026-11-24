"""Tests for the OOD gate and its pipeline integration."""
import glob
import os
import tempfile
import time

import numpy as np
import soundfile as sf

from audio_inference import CryAudioClassifier
from baby_profile import BabyProfile
from config import load_config
from ood import OODDetector
from pipeline import CryAgentPipeline

DONATE_DIR = os.path.expanduser(
    "~/.cache/huggingface/hub/datasets--Nooon--Donate_a_cry/snapshots/"
    "491487c4258287aa2be7cc179b7928768be7008d"
)


def test_synthetic_detector():
    rng = np.random.default_rng(7)
    centers = np.array([[0, 0, 0], [5, 5, 5]], dtype=float)
    X = np.vstack([
        rng.normal(centers[0], 0.4, size=(120, 3)),
        rng.normal(centers[1], 0.4, size=(120, 3)),
    ])
    y = np.array([0] * 120 + [1] * 120)
    groups = np.repeat(np.arange(24), 10)
    detector = OODDetector(calibration_percentile=85).fit(X, y, groups=groups)

    near = np.vstack([
        rng.normal(centers[0], 0.1, size=(20, 3)),
        rng.normal(centers[1], 0.1, size=(20, 3)),
    ])
    far = np.full((20, 3), 30.0)
    assert detector.is_ood(far).mean() >= 0.9
    assert detector.is_ood(near).mean() <= 0.25
    assert detector.score(far).min() > detector.score(near).max()
    print(f"  synthetic threshold={detector.threshold:.2f} "
          f"far={detector.is_ood(far).mean():.0%} "
          f"near={detector.is_ood(near).mean():.0%}")


def test_real_audio():
    classifier = CryAudioClassifier()
    assert classifier.ood is not None, "production OOD detector missing"
    files = sorted(glob.glob(os.path.join(DONATE_DIR, "*", "*.wav")))[:15]
    flagged = 0
    for path in files:
        result = classifier.classify_file(path)
        flagged += bool(result.get("ood"))
    assert flagged <= 6, f"{flagged}/15 in-distribution clips flagged"

    with tempfile.TemporaryDirectory() as tmp:
        noise_path = os.path.join(tmp, "noise.wav")
        rng = np.random.default_rng(0)
        sf.write(
            noise_path,
            rng.normal(0, 0.02, 32000).astype(np.float32),
            16000,
        )
        noise = classifier.classify_file(noise_path)
        assert noise.get("ood") is True, noise
        print(f"  in-distribution flagged={flagged}/15; noise "
              f"score={noise['ood_score']:.1f} > {noise['ood_threshold']:.1f}")


def _noise_file():
    path = os.path.join(tempfile.mkdtemp(), "noise.wav")
    rng = np.random.default_rng(0)
    sf.write(path, rng.normal(0, 0.02, 32000).astype(np.float32), 16000)
    return path


def test_pipeline_gate():
    pipeline = CryAgentPipeline(load_config())
    profile = BabyProfile(age_days=42)
    profile.last_feed_time = time.time() - 3600
    profile.last_sleep_end = time.time() - 2 * 3600
    noise_path = _noise_file()

    gated = pipeline.run(audio_path=noise_path, profile=profile)
    assert gated.cause == "unknown" and gated.model == "ood-gate"
    assert gated.audio_ood and not gated.trusted

    visual = pipeline.run(
        audio_path=noise_path, postures=["rooting"], profile=profile
    )
    assert visual.cause == "hungry" and visual.model == "fusion-nonaudio"
    assert visual.audio_ood and not visual.trusted
    print(f"  noise alone -> {gated.cause} via {gated.model}; "
          f"noise+rooting -> {visual.cause} via {visual.model}")


def test_normal_audio_trusted():
    path = sorted(glob.glob(os.path.join(DONATE_DIR, "hungry", "*.wav")))[0]
    pipeline = CryAgentPipeline(load_config())
    profile = BabyProfile(age_days=42)
    profile.last_feed_time = time.time() - 3 * 3600
    profile.last_sleep_end = time.time() - 2 * 3600
    result = pipeline.run(
        audio_path=path, postures=["rooting"], profile=profile
    )
    assert not result.audio_ood
    assert result.trusted
    assert result.cause != "unknown"
    print(f"  normal audio -> {result.cause} trusted={result.trusted}")


def test_session_gate():
    pipeline = CryAgentPipeline(load_config())
    pipeline.reset_session()
    profile = BabyProfile(age_days=42)
    profile.last_feed_time = time.time() - 3600
    profile.last_sleep_end = time.time() - 2 * 3600
    noise_path = _noise_file()
    for _ in range(8):
        pipeline.run(audio_path=noise_path, profile=profile)

    normal = sorted(glob.glob(os.path.join(DONATE_DIR, "hungry", "*.wav")))[0]
    result = pipeline.run(audio_path=normal, profile=profile)
    assert result.session_ood
    assert not result.trusted
    print(f"  after 8 odd clips, normal clip -> {result.cause} via "
          f"{result.model}, session_ood={result.session_ood}")

    pipeline.reset_session()
    result = pipeline.run(audio_path=normal, profile=profile)
    assert not result.session_ood and result.trusted
    print(f"  after reset -> trusted={result.trusted}")


if __name__ == "__main__":
    print("OOD GATE TESTS")
    print("-" * 60)
    test_synthetic_detector()
    test_real_audio()
    test_pipeline_gate()
    test_normal_audio_trusted()
    test_session_gate()
    print("-" * 60)
    print("All OOD tests passed.")
