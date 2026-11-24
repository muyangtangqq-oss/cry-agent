"""Live replay simulator — mimics real-time microphone input.

Takes a pre-recorded .wav file and feeds it through the pipeline
in 2-second segments, simulating real-time processing. Each segment
gets classified and fused with current context/vision/env signals.

Usage:
  uv run python live_replay.py <wav_file> [--postures p1,p2] [--temp 22] [--noise 40]
"""
import sys
import os
import time
import argparse
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import librosa

from config import load_config
from pipeline import CryAgentPipeline
from baby_profile import BabyProfile


def simulate_live_stream(wav_path, postures=None, face_state=None,
                         room_temp=None, noise_db=None, feed_gap=2.0,
                         sleep_gap=3.0, age_days=42):
    """Feed audio through pipeline in real-time segments."""
    cfg = load_config()
    pipeline = CryAgentPipeline(cfg)
    profile = BabyProfile(age_days=age_days)
    profile.last_feed_time = time.time() - feed_gap * 3600
    profile.last_sleep_end = time.time() - sleep_gap * 3600

    # Load full audio
    y, sr = librosa.load(wav_path, sr=16000)
    total_duration = len(y) / sr
    seg_len = int(sr * 2.0)  # 2-second segments
    n_segments = max(1, len(y) // seg_len)

    print("=" * 70)
    print("LIVE REPLAY SIMULATION")
    print("=" * 70)
    print(f"File: {os.path.basename(wav_path)}")
    print(f"Duration: {total_duration:.1f}s | Segments: {n_segments} (2s each)")
    print(f"Context: feed_gap={feed_gap}h, sleep_gap={sleep_gap}h, age={age_days}d")
    if postures:
        print(f"Vision: postures={postures}, face={face_state or 'normal'}")
    if room_temp:
        print(f"Env: temp={room_temp}°C, noise={noise_db}dB")
    print()
    print(f"{'Seg':>4s} {'Time':>6s} {'Audio':>12s} {'Conf':>6s} "
          f"{'Fused':>12s} {'Conf':>6s} {'Model':>10s} {'Final':>12s} "
          f"{'Override':>9s} {'OOD':>4s} {'Lat':>5s}")
    print("-" * 101)

    all_results = []

    for seg_idx in range(n_segments):
        start = seg_idx * seg_len
        end = min(start + seg_len, len(y))
        seg_audio = y[start:end]

        if len(seg_audio) < sr * 0.3:
            continue

        # Write segment to temp file for pipeline
        tmp_path = f"/tmp/cry_seg_{seg_idx}.wav"
        import soundfile as sf
        sf.write(tmp_path, seg_audio, 16000)

        # Run pipeline on this segment
        result = pipeline.run(
            audio_path=tmp_path,
            postures=postures,
            face_state=face_state,
            room_temp=room_temp,
            ambient_noise_db=noise_db,
            profile=profile,
        )

        seg_time = start / sr
        all_results.append({
            "seg": seg_idx,
            "time": seg_time,
            "result": result,
        })

        override = "YES" if result.critical_override else ""
        ood = "OOD" if (result.audio_ood or result.session_ood) else ""
        print(f"{seg_idx:4d} {seg_time:5.1f}s {result.audio_cause:>12s} "
              f"{result.audio_confidence:>5.0%} {result.fused_cause:>12s} "
              f"{result.fused_confidence:>5.0%} {result.model:>10s} "
              f"{result.cause:>12s} {override:>9s} {ood:>4s} "
              f"{result.latency_ms:>4.0f}ms")

        # Simulate real-time pacing (but faster for demo)
        time.sleep(0.1)

        os.unlink(tmp_path)

    # Aggregate result across all segments
    print(f"\n{'='*70}")
    print("AGGREGATE RESULT")
    print(f"{'='*70}")

    if not all_results:
        print("No segments processed.")
        return

    # Vote across all segments
    from collections import Counter
    causes = [r["result"].cause for r in all_results]
    cause_counts = Counter(causes)
    final_cause = cause_counts.most_common(1)[0][0]

    # Average confidence for winning cause
    winning_confs = [
        r["result"].confidence for r in all_results
        if r["result"].cause == final_cause
    ]
    avg_conf = sum(winning_confs) / len(winning_confs) if winning_confs else 0

    avg_latency = sum(r["result"].latency_ms for r in all_results) / len(all_results)

    print(f"Segments processed:   {len(all_results)}")
    print(f"Cause distribution:   {dict(cause_counts)}")
    print(f"Final cause:          {final_cause} (avg confidence {avg_conf:.0%})")
    print(f"Average latency:      {avg_latency:.0f}ms/segment")
    print(f"Models used:          {Counter(r['result'].model for r in all_results)}")
    print(f"Critical overrides:   {sum(1 for r in all_results if r['result'].critical_override)}")
    print(f"Escalated segments:   {sum(1 for r in all_results if r['result'].escalated)}")
    print(f"OOD / untrusted:      "
          f"{sum(1 for r in all_results if r['result'].audio_ood or r['result'].session_ood)}"
          f" / {sum(1 for r in all_results if not r['result'].trusted)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Live replay simulation")
    parser.add_argument("wav_file", help="Path to .wav file")
    parser.add_argument("--postures", type=str, default="",
                        help="Comma-separated postures (e.g. rooting,sucking_fingers)")
    parser.add_argument("--face", type=str, default=None,
                        help="Face state: red, pale, or none")
    parser.add_argument("--temp", type=float, default=None,
                        help="Room temperature")
    parser.add_argument("--noise", type=float, default=None,
                        help="Ambient noise dB")
    parser.add_argument("--feed-gap", type=float, default=2.0,
                        help="Hours since last feed")
    parser.add_argument("--sleep-gap", type=float, default=3.0,
                        help="Hours since last sleep")
    parser.add_argument("--age", type=int, default=42,
                        help="Baby age in days")
    args = parser.parse_args()

    postures = [p.strip() for p in args.postures.split(",") if p.strip()]
    simulate_live_stream(
        args.wav_file,
        postures=postures or None,
        face_state=args.face,
        room_temp=args.temp,
        noise_db=args.noise,
        feed_gap=args.feed_gap,
        sleep_gap=args.sleep_gap,
        age_days=args.age,
    )
