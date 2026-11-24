"""Regression + extension tests for the configuration-driven engine.

1. Baby YAML must reproduce the original hard-coded extractors/fuser.
2. Elderly and pet YAMLs must run through the same pipeline unchanged.
"""
import time

from baby_profile import BabyProfile
from config import load_config
from multimodal_signals import (
    ContextSignalExtractor,
    EnvSignalExtractor,
    MultimodalFuser,
    Signal,
    VisionSignalExtractor,
)
from scenario_engine import Scenario, list_scenarios

NOW = time.struct_time((2026, 11, 24, 21, 0, 0, 1, 328, 0))

SCENARIOS = [
    {"postures": ["rooting", "sucking_fingers"], "face": None,
     "temp": 23, "noise": 40, "feed_gap": 3.0, "sleep_gap": 2.0,
     "audio": ("hungry", 0.98), "expected": "hungry"},
    {"postures": ["rubbing_eyes", "yawning"], "face": None,
     "temp": 22, "noise": 35, "feed_gap": 1.0, "sleep_gap": 4.5,
     "audio": ("sleepy", 0.60), "expected": "sleepy"},
    {"postures": ["arching_back", "clenched_fists"], "face": None,
     "temp": 24, "noise": 50, "feed_gap": 0.8, "sleep_gap": 1.5,
     "audio": ("hungry", 0.55), "expected": "pain"},
    {"postures": ["arms_spread"], "face": None,
     "temp": 22, "noise": 90, "feed_gap": 1.5, "sleep_gap": 2.0,
     "audio": ("startle", 0.40), "expected": "startle"},
    {"postures": ["fetal_position"], "face": "pale",
     "temp": 17, "noise": 30, "feed_gap": 1.0, "sleep_gap": 2.0,
     "audio": ("discomfort", 0.50), "expected": "temperature"},
    {"postures": ["reaching_up", "looking_around"], "face": None,
     "temp": 23, "noise": 35, "feed_gap": 1.0, "sleep_gap": 2.0,
     "audio": ("hungry", 0.98), "expected": "lonely"},
]


def _signals(extractors, scenario):
    vision_ext, context_ext, env_ext = extractors
    profile = BabyProfile(age_days=42)
    profile.last_feed_time = time.time() - scenario["feed_gap"] * 3600
    profile.last_sleep_end = time.time() - scenario["sleep_gap"] * 3600

    audio = Signal("cry_classification", scenario["audio"][0],
                   scenario["audio"][1], "audio")
    vision = vision_ext.extract(scenario["postures"], scenario["face"])
    context = context_ext.extract(profile, now=NOW)
    env = env_ext.extract(scenario["temp"], None, scenario["noise"], now=NOW)
    return [audio, vision, context, env]


def test_scenarios_load():
    names = list(list_scenarios())
    assert {"baby_cry", "elderly_care", "pet_behavior"} <= set(names), names
    for name in names:
        scenario = Scenario.load(name)
        assert scenario.classes, f"{name} has no audio classes"
        print(f"  loaded {name:<15s} classes={len(scenario.classes):>2d} "
              f"direct={scenario.threshold('direct_output', 0.80):.2f}")


def test_baby_equivalence():
    baby = Scenario.load("baby_cry")
    legacy_extractors = (
        VisionSignalExtractor(),
        ContextSignalExtractor(),
        EnvSignalExtractor(),
    )
    config_extractors = (
        baby.build_vision_extractor(),
        baby.build_context_extractor(),
        baby.build_env_extractor(),
    )
    legacy_fuser = MultimodalFuser()
    config_fuser = baby.build_fuser()

    for idx, scenario in enumerate(SCENARIOS, 1):
        legacy = _signals(legacy_extractors, scenario)
        configured = _signals(config_extractors, scenario)

        for left, right in zip(legacy, configured):
            assert (left.value, left.confidence) == (right.value, right.confidence), (
                idx, left, right
            )

        legacy_result = legacy_fuser.fuse(legacy)
        config_result = config_fuser.fuse(configured)
        for key in ("fused_cause", "fused_confidence", "escalate",
                    "critical_override"):
            assert legacy_result[key] == config_result[key], (
                idx, key, legacy_result[key], config_result[key]
            )
        assert config_result["fused_cause"] == scenario["expected"], (
            idx, config_result["fused_cause"]
        )
        print(f"  baby scenario {idx}: {config_result['fused_cause']} "
              f"{config_result['fused_confidence']:.0%} == legacy")


def test_elderly_pipeline():
    from pipeline import CryAgentPipeline

    pipeline = CryAgentPipeline(load_config(), "elderly_care")
    result = pipeline.run(
        postures=["lying_down"],
        env_values={"floor_pressure": 60},
    )
    assert result.cause == "fall", result
    assert result.critical_override, result
    print(f"  elderly fall: {result.cause} ({result.confidence:.0%}) "
          f"via {result.model}")


def test_pet_fusion():
    pet = Scenario.load("pet_behavior")
    vision = pet.build_vision_extractor().extract(["tail_tucked"])
    context = pet.build_context_extractor().extract(
        values={"hours_alone": 5}, now=NOW
    )
    fused = pet.build_fuser().fuse([
        Signal("cry_classification", "unknown", 0.0, "audio"),
        vision, context,
    ])
    assert fused["fused_cause"] == "fear", fused
    assert fused["critical_override"], fused
    print(f"  pet fear: {fused['fused_cause']} "
          f"({fused['fused_confidence']:.0%}) override={fused['critical_override']}")


if __name__ == "__main__":
    print("SCENARIO ENGINE TESTS")
    print("-" * 60)
    print("[1] scenario configs")
    test_scenarios_load()
    print("[2] baby YAML == legacy hard-coded behavior")
    test_baby_equivalence()
    print("[3] elderly scenario through the full pipeline")
    test_elderly_pipeline()
    print("[4] pet scenario fusion")
    test_pet_fusion()
    print("-" * 60)
    print("All scenario engine tests passed.")
