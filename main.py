"""Scenario CLI for cry-agent.

Examples:
  python main.py --list
  python main.py --describe elderly_care
  python main.py --audio sample.wav --postures rooting,sucking_fingers
  python main.py --scenario elderly_care --sensor floor_pressure=60
"""
import argparse
import json
import time

from config import load_config
from scenario_engine import Scenario, list_scenarios


def _pairs(items):
    values = {}
    for item in items or []:
        key, _, raw = item.partition("=")
        if not key or not _:
            raise SystemExit(f"Expected key=value, got: {item}")
        try:
            values[key] = float(raw)
        except ValueError:
            values[key] = raw
    return values


def main():
    parser = argparse.ArgumentParser(description="cry-agent scenario CLI")
    parser.add_argument("--list", action="store_true", help="list scenarios")
    parser.add_argument("--describe", metavar="NAME", help="print scenario summary")
    parser.add_argument("--scenario", default="baby_cry", help="scenario name or YAML path")
    parser.add_argument("--audio", help="path to a .wav file to classify")
    parser.add_argument("--postures", default="", help="comma-separated posture labels")
    parser.add_argument("--face", help="face state, e.g. red / pale")
    parser.add_argument("--temp", type=float, help="room temperature in Celsius")
    parser.add_argument("--noise", type=float, help="ambient noise in dB")
    parser.add_argument("--feed-gap", type=float, help="hours since last feed")
    parser.add_argument("--sleep-gap", type=float, help="hours since last sleep")
    parser.add_argument("--context-field", action="append", metavar="K=V",
                        help="extra context value declared by the scenario")
    parser.add_argument("--sensor", action="append", metavar="K=V",
                        help="extra env sensor value declared by the scenario")
    parser.add_argument("--json", action="store_true", help="JSON output")
    args = parser.parse_args()

    if args.list:
        for name, path in list_scenarios().items():
            print(f"{name:<20s} {path}")
        return

    scenario = Scenario.load(args.describe or args.scenario)
    if args.describe:
        print(scenario.summary())
        return

    if not args.audio and not args.postures and not args.sensor:
        print(scenario.summary())
        print("\nPass --audio or --postures/--sensor to run a diagnosis.")
        return

    from pipeline import CryAgentPipeline
    from baby_profile import BabyProfile

    pipeline = CryAgentPipeline(load_config(), scenario)
    profile = BabyProfile(age_days=scenario.default_age_days)
    if args.feed_gap is not None:
        profile.last_feed_time = time.time() - args.feed_gap * 3600
    if args.sleep_gap is not None:
        profile.last_sleep_end = time.time() - args.sleep_gap * 3600

    result = pipeline.run(
        audio_path=args.audio,
        postures=[p.strip() for p in args.postures.split(",") if p.strip()],
        face_state=args.face,
        room_temp=args.temp,
        ambient_noise_db=args.noise,
        profile=profile,
        context_values=_pairs(args.context_field),
        env_values=_pairs(args.sensor),
    )

    if args.json:
        print(json.dumps(result.__dict__, indent=2, default=str))
        return

    print(f"Scenario: {result.scenario}")
    print(f"Cause:    {result.cause} ({result.confidence:.0%}) via {result.model}")
    print(f"Fused:    {result.fused_cause} ({result.fused_confidence:.0%})")
    print(f"Audio:    {result.audio_cause} ({result.audio_confidence:.0%})")
    print(f"Vision:   {result.vision_cause} ({result.vision_confidence:.0%})")
    print(f"Latency:  {result.latency_ms:.0f} ms")
    print(f"Escalate: {result.escalated}  override: {result.critical_override}")


if __name__ == "__main__":
    main()
