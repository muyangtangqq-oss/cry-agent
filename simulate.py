"""Simulation harness — tests the layered judge with synthetic scenarios.

This replaces real audio/posture input with synthetic signal sets,
letting us validate the architecture without a real baby or hardware.
"""
import json
import time
import os

from config import load_config
from judge import LayeredJudge
from baby_profile import BabyProfile, CryEvent

# --- Test scenarios ---
SCENARIOS = [
    {
        "name": "hungry_after_long_gap",
        "signals": {
            "cry_pattern": "rhythmic medium pitch short bursts",
            "posture": "rooting, sucking fingers",
            "context_hint": "last feed 3 hours ago",
        },
        "expected": "hungry",
        "feed_gap_hours": 3.0,
    },
    {
        "name": "sleepy_rubbing_eyes",
        "signals": {
            "cry_pattern": "whimpering low energy",
            "posture": "rubbing eyes, yawning",
            "context_hint": "last sleep 4 hours ago",
        },
        "expected": "sleepy",
        "feed_gap_hours": 1.0,
    },
    {
        "name": "diaper_wet",
        "signals": {
            "cry_pattern": "fussing intermittent",
            "posture": "squirming, kicking legs",
            "context_hint": "last diaper change unknown",
        },
        "expected": "discomfort",
        "feed_gap_hours": 1.5,
    },
    {
        "name": "colic_pain",
        "signals": {
            "cry_pattern": "sharp high pitch sustained",
            "posture": "arching back, clenched fists",
            "context_hint": "evening time, last feed 30min ago",
        },
        "expected": "pain",
        "feed_gap_hours": 0.5,
    },
    {
        "name": "startle_loud_noise",
        "signals": {
            "cry_pattern": "sudden scream",
            "posture": "startled arms spread",
            "context_hint": "loud noise just occurred",
        },
        "expected": "startle",
        "feed_gap_hours": 1.0,
    },
    {
        "name": "ambiguous_low_confidence",
        "signals": {
            "cry_pattern": "crying",
            "posture": "moving",
            "context_hint": "no clear signals",
        },
        "expected": None,  # should be low confidence
        "feed_gap_hours": 2.0,
    },
]


def run_simulation():
    cfg = load_config()
    profile_path = os.path.expanduser("~/.config/cry-agent/profile.json")
    profile = BabyProfile.load(profile_path)
    judge = LayeredJudge(cfg)

    print("=" * 70)
    print("CRY-AGENT LAYERED JUDGE SIMULATION")
    print("=" * 70)
    print(f"Tachyone threshold: high={cfg.tachyone_high} low={cfg.tachyone_low}")
    print(f"Jev API key: {'configured' if cfg.typesafe_api_key else 'NOT SET'}")
    print()

    results = []
    for i, scenario in enumerate(SCENARIOS):
        # Set up profile for this scenario
        profile.last_feed_time = time.time() - scenario["feed_gap_hours"] * 3600

        # Run judgment
        result = judge.judge(scenario["signals"], profile)
        results.append((scenario, result))

        # Print result
        print(f"[{i+1}/{len(SCENARIOS)}] {scenario['name']}")
        print(f"  Expected: {scenario['expected'] or '(low confidence)'}")
        print(f"  Tachyone: {result.cause} ({result.tachyone_confidence})")
        if result.jev_confidence is not None:
            print(f"  Jev:      {result.jev_confidence}")
        print(f"  Final:    {result.cause} ({result.confidence}) via {result.model}")
        print(f"  Escalated: {result.escalated}  Latency: {result.latency_ms}ms")
        if result.alternatives:
            alts = ", ".join(
                f"{a['answer']}({a['confidence']})" for a in result.alternatives[:2]
            )
            print(f"  Alternatives: {alts}")
        print()

    # Summary
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    correct = 0
    escalated = 0
    for scenario, result in results:
        if scenario["expected"] and result.cause == scenario["expected"]:
            correct += 1
        if result.escalated:
            escalated += 1
    total = len(results)
    print(f"Accuracy: {correct}/{total} ({correct/total*100:.0f}%)")
    print(f"Escalated to Jev: {escalated}/{total}")
    print(f"Avg latency: {sum(r.latency_ms for _, r in results)/total:.0f}ms")
    print()

    return results


if __name__ == "__main__":
    run_simulation()
