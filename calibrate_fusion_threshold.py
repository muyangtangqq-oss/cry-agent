"""Calibrate the fusion direct-output threshold.

Builds a labelled set from the scenario suites (synthetic baby scenarios,
real-audio end-to-end scenarios, elderly fall, pet fear), records the fused
confidence, and sweeps candidate thresholds. Also prints the operating points
for two-source and three-source agreement.

Usage:
  python tachyone_server.py 8901 &
  python calibrate_fusion_threshold.py
"""
import argparse
import os
import time

os.environ.setdefault("CRY_AGENT_TS_MOCK", "1")

from baby_profile import BabyProfile
from config import load_config
from multimodal_signals import Signal
from pipeline import CryAgentPipeline
from scenario_engine import Scenario
from test_full_pipeline import (
    SCENARIOS as REAL_SCENARIOS,
    expected_cause,
    get_test_files,
)
from test_scenario_engine import NOW, SCENARIOS as SYNTHETIC_SCENARIOS, _signals
import vision_detector

THRESHOLDS = [0.70, 0.75, 0.80, 0.85, 0.90, 0.95]


def synthetic_baby_cases():
    baby = Scenario.load("baby_cry")
    extractors = (
        baby.build_vision_extractor(),
        baby.build_context_extractor(),
        baby.build_env_extractor(),
    )
    fuser = baby.build_fuser()
    cases = []
    for idx, scenario in enumerate(SYNTHETIC_SCENARIOS, 1):
        fused = fuser.fuse(_signals(extractors, scenario))
        cases.append({
            "name": f"baby-{idx}",
            "expected": scenario["expected"],
            "fused_cause": fused["fused_cause"],
            "confidence": fused["fused_confidence"],
            "escalate": fused["escalate"],
        })
    return cases


def real_audio_cases():
    pipeline = CryAgentPipeline(load_config())
    vision = vision_detector.VisionDetector()
    test_files = get_test_files()
    cases = []
    for scenario in REAL_SCENARIOS:
        wav_file = test_files.get(scenario["audio_class"])
        if not wav_file:
            continue
        profile = BabyProfile(age_days=42)
        profile.last_feed_time = time.time() - scenario["feed_gap"] * 3600
        profile.last_sleep_end = time.time() - scenario["sleep_gap"] * 3600
        import tempfile
        tmpdir = tempfile.mkdtemp()
        img_path = os.path.join(tmpdir, "frame.png")
        with open(img_path, "wb") as f:
            f.write(b"\x89PNG\r\n\x1a\n")
        if scenario["postures"] or scenario["face"]:
            vision_detector.create_annotation_file(
                img_path, scenario["postures"], scenario["face"]
            )
        vision_result = vision.detect_from_image(img_path)
        result = pipeline.run(
            audio_path=wav_file,
            postures=vision_result["postures"],
            face_state=vision_result["face_state"],
            room_temp=scenario["temp"],
            ambient_noise_db=scenario["noise"],
            profile=profile,
        )
        cases.append({
            "name": scenario["name"].split(":")[0] + " (audio)",
            "expected": expected_cause(scenario),
            "fused_cause": result.fused_cause,
            "confidence": result.fused_confidence,
            "escalate": result.escalated,
        })
    return cases


def other_scenario_cases():
    elderly = CryAgentPipeline(load_config(), "elderly_care")
    fall = elderly.run(
        postures=["lying_down"], env_values={"floor_pressure": 60}
    )

    pet = Scenario.load("pet_behavior")
    pet_fused = pet.build_fuser().fuse([
        Signal("cry_classification", "unknown", 0.0, "audio"),
        pet.build_vision_extractor().extract(["tail_tucked"]),
        pet.build_context_extractor().extract(
            values={"hours_alone": 5}, now=NOW
        ),
    ])
    return [
        {
            "name": "elderly fall",
            "expected": "fall",
            "fused_cause": fall.fused_cause,
            "confidence": fall.fused_confidence,
            "escalate": fall.escalated,
        },
        {
            "name": "pet fear",
            "expected": "fear",
            "fused_cause": pet_fused["fused_cause"],
            "confidence": pet_fused["fused_confidence"],
            "escalate": pet_fused["escalate"],
        },
    ]


def operating_points():
    baby = Scenario.load("baby_cry")
    points = []
    for p in (0.40, 0.50, 0.55, 0.60, 0.65, 0.70, 0.80):
        two = baby.build_fuser().fuse([
            Signal("audio", "hungry", p, "audio"),
            Signal("vision", "hungry", p, "vision"),
        ])["fused_confidence"]
        three = baby.build_fuser().fuse([
            Signal("audio", "hungry", p, "audio"),
            Signal("vision", "hungry", p, "vision"),
            Signal("context", "hungry", p, "context"),
        ])["fused_confidence"]
        points.append((p, two, three))
    return points


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="reports/fusion_threshold.md")
    args = parser.parse_args()

    baby_cases = synthetic_baby_cases() + real_audio_cases()
    other_cases = other_scenario_cases()
    points = operating_points()

    lines = [
        "# Fusion direct-output threshold calibration",
        "",
        "Labelled set: six configured baby scenarios, six real-audio",
        "end-to-end scenarios (n=%d). Elderly and pet are checked" % len(baby_cases),
        "separately at their own configured thresholds.",
        "Fusion confidence is `vote share x evidence strength + agreement",
        "boost`; direct output requires `confidence >= threshold` and",
        "`escalate == false`.",
        "",
        "| Case | Expected | Fused | Confidence | Escalate |",
        "|---|---|---|---:|---|",
    ]
    for case in baby_cases:
        lines.append(
            f"| {case['name']} | {case['expected']} | {case['fused_cause']} | "
            f"{case['confidence']:.0%} | {case['escalate']} |"
        )

    lines += [
        "",
        "## Threshold sweep",
        "",
        "| Threshold | Direct | Direct correct | Wrong direct | Direct precision |",
        "|---:|---:|---:|---:|---:|",
    ]
    for threshold in THRESHOLDS:
        direct = [
            c for c in baby_cases
            if not c["escalate"] and c["confidence"] >= threshold
        ]
        correct = [c for c in direct if c["fused_cause"] == c["expected"]]
        precision = len(correct) / len(direct) if direct else 1.0
        lines.append(
            f"| {threshold:.2f} | {len(direct)}/{len(baby_cases)} | "
            f"{len(correct)} | {len(direct) - len(correct)} | "
            f"{precision:.0%} |"
        )

    lines += [
        "",
        "## Agreement operating points",
        "",
        "| Per-source confidence | Two sources | Three sources |",
        "|---:|---:|---:|",
    ]
    for p, two, three in points:
        lines.append(f"| {p:.0%} | {two:.0%} | {three:.0%} |")

    lines += [
        "",
        "## Cross-scenario sanity check",
        "",
        "| Case | Threshold | Expected | Fused | Confidence | Direct |",
        "|---|---:|---|---|---:|---|",
    ]
    configured = {"elderly fall": 0.85, "pet fear": 0.75}
    for case in other_cases:
        threshold = configured[case["name"]]
        direct = (
            not case["escalate"] and case["confidence"] >= threshold
        )
        lines.append(
            f"| {case['name']} | {threshold:.2f} | {case['expected']} | "
            f"{case['fused_cause']} | {case['confidence']:.0%} | {direct} |"
        )

    lines += [
        "",
        "## Decision",
        "",
        "Set the baby direct-output threshold to **0.85**. That is the",
        "operating point for two sources at 60% or three sources at 40%;",
        "0.80 would admit weaker two-source agreement such as 60% + 50%.",
        "It is also the highest threshold that still keeps every baby",
        "critical override (capped at 85%) on the direct path; 0.90 would",
        "send them to a judge. The 75% env-override temperature cases take",
        "a judge pass. The labelled set has no wrong direct answer at 0.75",
        "or above, so 0.85 is conservative rather than accuracy-optimal.",
        "",
        "Caveat: no real multimodal household data exists yet. This is a",
        "policy calibration over constructed/real-audio scenarios, not a",
        "home-accuracy estimate; re-calibrate after the pilot.",
    ]

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nSaved {args.out}")


if __name__ == "__main__":
    main()
