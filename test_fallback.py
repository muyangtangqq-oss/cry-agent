"""Test Jev fallback chain — forces low-confidence scenarios.

Constructs signals where audio confidence is low, so the judgment
must escalate through Tachyone → Jev to get an answer.
"""
import time
from config import load_config
from judge import LayeredJudge
from baby_profile import BabyProfile

# Scenarios designed to trigger fallback at different tiers
SCENARIOS = [
    {
        "name": "low_conf_hungry_context",
        "signals": {
            "cry_pattern": "uncertain mixed pattern",
            "audio_top_cause": "hungry",
            "audio_confidence": 0.45,  # below 0.80, triggers Tachyone
            "audio_probabilities": {"hungry": 0.45, "discomfort": 0.30, "sleepy": 0.25},
        },
        "feed_gap": 3.0,  # supports hungry
        "sleep_gap": 2.0,
        "expected": "hungry",
    },
    {
        "name": "low_conf_sleepy_context",
        "signals": {
            "cry_pattern": "whimpering",
            "audio_top_cause": "sleepy",
            "audio_confidence": 0.50,
            "audio_probabilities": {"sleepy": 0.50, "hungry": 0.35, "discomfort": 0.15},
        },
        "feed_gap": 1.0,  # not hungry
        "sleep_gap": 4.5,  # should be sleepy
        "expected": "sleepy",
    },
    {
        "name": "ambiguous_equal_split",
        "signals": {
            "cry_pattern": "crying",
            "audio_top_cause": "discomfort",
            "audio_confidence": 0.35,  # very low, should go to Jev
            "audio_probabilities": {"discomfort": 0.35, "hungry": 0.33, "pain": 0.32},
        },
        "feed_gap": 1.5,
        "sleep_gap": 2.0,
        "expected": "discomfort",
    },
    {
        "name": "post_feed_burp",
        "signals": {
            "cry_pattern": "brief fussing",
            "audio_top_cause": "burp",
            "audio_confidence": 0.42,
            "audio_probabilities": {"burp": 0.42, "hungry": 0.38, "discomfort": 0.20},
        },
        "feed_gap": 0.5,  # just fed — burp more likely than hungry
        "sleep_gap": 1.0,
        "expected": "burp",
    },
    {
        "name": "evening_colic",
        "signals": {
            "cry_pattern": "sharp sustained",
            "audio_top_cause": "pain",
            "audio_confidence": 0.48,
            "audio_probabilities": {"pain": 0.48, "hungry": 0.40, "discomfort": 0.12},
        },
        "feed_gap": 0.8,
        "sleep_gap": 1.5,
        "expected": "pain",
    },
    {
        "name": "very_low_all_unknown",
        "signals": {
            "cry_pattern": "unclear",
            "audio_top_cause": "discomfort",
            "audio_confidence": 0.20,  # extremely low
            "audio_probabilities": {"discomfort": 0.20, "hungry": 0.20, "sleepy": 0.20, "pain": 0.20, "burp": 0.20},
        },
        "feed_gap": 2.0,
        "sleep_gap": 2.0,
        "expected": None,  # should be uncertain
    },
]


def run():
    cfg = load_config()
    judge = LayeredJudge(cfg)
    profile = BabyProfile.load(
        time.strftime("~/.config/cry-agent/profile.json")
        if False else "~/.config/cry-agent/profile.json"
    )
    # But expand the path
    import os
    profile = BabyProfile.load(os.path.expanduser("~/.config/cry-agent/profile.json"))

    print("=" * 70)
    print("FALLBACK CHAIN VALIDATION")
    print("=" * 70)
    print(f"Audio direct threshold: 0.80")
    print(f"Tachyone high threshold: {cfg.tachyone_high}")
    print(f"Tachyone low threshold: {cfg.tachyone_low}")
    print()

    results = []
    for i, scenario in enumerate(SCENARIOS):
        profile.last_feed_time = time.time() - scenario["feed_gap"] * 3600
        profile.last_sleep_end = time.time() - scenario["sleep_gap"] * 3600

        result = judge.judge(scenario["signals"], profile)

        chain = []
        if result.audio_confidence >= 0.80:
            chain.append("audio(direct)")
        else:
            chain.append(f"audio({result.audio_confidence:.0%})")
            if result.tachyone_confidence is not None:
                chain.append(f"tachyone({result.tachyone_confidence:.0%})")
            if result.jev_confidence is not None:
                chain.append(f"jev({result.jev_confidence:.0%})")

        chain_str = " → ".join(chain)
        correct = "—" if scenario["expected"] is None else (
            "✓" if result.cause == scenario["expected"] else "✗"
        )

        print(f"[{i+1}/{len(SCENARIOS)}] {scenario['name']}")
        print(f"  Expected: {scenario['expected'] or '(uncertain)'}")
        print(f"  Chain:    {chain_str}")
        print(f"  Final:    {result.cause} ({result.confidence:.0%}) via {result.model}  {correct}")
        if result.alternatives:
            alts = result.alternatives
            if isinstance(alts, list):
                alt_str = ", ".join(f"{a['answer']}({a['confidence']:.0%})" for a in alts[:3])
            elif isinstance(alts, dict):
                alt_str = ", ".join(f"{k}({v:.0%})" for k, v in
                                    sorted(alts.items(), key=lambda x: -x[1])[:3])
            else:
                alt_str = str(alts)[:80]
            print(f"  Alts:     {alt_str}")
        print(f"  Latency:  {result.latency_ms:.0f}ms")
        print()

        results.append((scenario, result, correct))

    # Summary
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    evaluable = [(s, r, c) for s, r, c in results if s["expected"] is not None]
    correct_count = sum(1 for _, _, c in evaluable if c == "✓")
    total = len(evaluable)
    jev_used = sum(1 for _, r, _ in results if r.jev_confidence is not None)
    tachyone_used = sum(1 for _, r, _ in results if r.tachyone_confidence is not None and r.audio_confidence < 0.80)

    print(f"Scenarios: {len(results)} ({total} evaluable)")
    print(f"Tachyone invoked: {tachyone_used}/{len(results)}")
    print(f"Jev invoked: {jev_used}/{len(results)}")
    print(f"Accuracy on uncertain scenarios: {correct_count}/{total} ({correct_count/total*100:.0f}%)")

    # Show which model won each
    print(f"\nModel selection:")
    model_counts = {}
    for _, r, _ in results:
        model_counts[r.model] = model_counts.get(r.model, 0) + 1
    for model, count in sorted(model_counts.items()):
        print(f"  {model:15s}: {count}")


if __name__ == "__main__":
    run()
