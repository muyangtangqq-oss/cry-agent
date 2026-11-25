"""Fusion confidence calibration tests.

Guards the normalization fix: a cause is not 100% confident just because it
is the only cause with evidence. Confidence must combine vote share with
absolute evidence mass, and multi-source agreement must still boost it.
"""
from multimodal_signals import MultimodalFuser, Signal
from scenario_engine import Scenario


def fuse(*signals):
    return MultimodalFuser().fuse(list(signals))


def test_single_weak_source_is_not_certain():
    result = fuse(Signal("audio", "hungry", 0.20, "audio"))
    assert result["fused_cause"] == "hungry"
    assert result["fused_confidence"] == 0.133, result
    assert result["escalate"] is False  # one cause has no top-2 margin


def test_single_strong_source_stays_below_direct_threshold():
    result = fuse(Signal("audio", "hungry", 1.0, "audio"))
    assert result["fused_confidence"] == 0.667, result


def test_two_strong_agreeing_sources_are_direct():
    result = fuse(
        Signal("audio", "pain", 0.90, "audio"),
        Signal("vision", "pain", 0.95, "vision"),
    )
    assert result["fused_cause"] == "pain"
    assert result["fused_confidence"] == 0.95, result


def test_two_weak_agreeing_sources_stay_low():
    result = fuse(
        Signal("audio", "hungry", 0.20, "audio"),
        Signal("vision", "hungry", 0.20, "vision"),
    )
    assert result["fused_confidence"] == 0.383, result


def test_disagreement_is_attenuated():
    result = fuse(
        Signal("audio", "hungry", 0.90, "audio"),
        Signal("vision", "discomfort", 0.60, "vision"),
    )
    assert result["fused_cause"] == "hungry"
    assert result["fused_confidence"] == 0.600, result

    split = fuse(
        Signal("audio", "hungry", 0.50, "audio"),
        Signal("vision", "discomfort", 0.50, "vision"),
    )
    assert split["fused_confidence"] == 0.333, split


def test_three_weak_sources_are_moderate():
    result = fuse(
        Signal("audio", "hungry", 0.30, "audio"),
        Signal("vision", "hungry", 0.30, "vision"),
        Signal("context", "hungry", 0.30, "context"),
    )
    assert result["fused_confidence"] == 0.70, result


def test_confidence_is_bounded():
    worst = fuse(
        Signal("audio", "hungry", 1.0, "audio"),
        Signal("vision", "hungry", 1.0, "vision"),
        Signal("context", "hungry", 1.0, "context"),
        Signal("env", "hungry", 1.0, "env"),
    )
    assert 0.0 <= worst["fused_confidence"] <= 0.98, worst


def test_direct_threshold_operating_points():
    scenario = Scenario.load("baby_cry")
    threshold = scenario.threshold("direct_output", 0.85)
    assert threshold == 0.85, threshold

    two_sources = fuse(
        Signal("audio", "hungry", 0.60, "audio"),
        Signal("vision", "hungry", 0.60, "vision"),
    )
    assert two_sources["fused_confidence"] == threshold, two_sources

    three_sources = fuse(
        Signal("audio", "hungry", 0.40, "audio"),
        Signal("vision", "hungry", 0.40, "vision"),
        Signal("context", "hungry", 0.40, "context"),
    )
    assert three_sources["fused_confidence"] == threshold, three_sources


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"  ok {name}")
    print("Fusion calibration tests passed.")
