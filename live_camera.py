"""Live camera + microphone demo for Jetson/Pi deployment.

Combines:
  - Real-time camera input → MediaPipe BlazePose posture detection
  - Real-time microphone input → audio cry classification
  - Environment sensors → temperature + noise
  - Multimodal fusion → layered judge → final output

Runs continuously, processing one cycle every N seconds.

Usage on Jetson:
  python3 live_camera.py [--camera 0] [--mic-device 2] [--interval 3]

Usage without camera (replay mode):
  python3 live_camera.py --replay-audio <wav> [--postures rooting]
"""
import sys
import os
import time
import argparse
import threading
import warnings
warnings.filterwarnings("ignore")

import numpy as np

from config import load_config
from pipeline import CryAgentPipeline
from baby_profile import BabyProfile
from vision_mediapipe import create_vision_detector, CameraStream, HAS_MEDIAPIPE


class MicrophoneStream:
    """Real-time microphone input using sounddevice.

    Falls back to file replay if no microphone available.
    """
    def __init__(self, device=None, sample_rate=16000, segment_duration=2.0):
        self.device = device
        self.sample_rate = sample_rate
        self.segment_duration = segment_duration
        self.segment_samples = int(sample_rate * segment_duration)
        self.buffer = []
        self.is_recording = False
        self._stream = None

        try:
            import sounddevice as sd
            self.has_mic = True
            self.sd = sd
        except ImportError:
            self.has_mic = False
            print("[mic] sounddevice not installed — use --replay-audio")

    def start(self):
        if not self.has_mic:
            return
        self.is_recording = True
        self.buffer = []
        self._stream = self.sd.InputStream(
            device=self.device,
            samplerate=self.sample_rate,
            channels=1,
            dtype='float32',
            callback=self._callback,
        )
        self._stream.start()

    def _callback(self, indata, frames, time_info, status):
        if self.is_recording:
            self.buffer.extend(indata[:, 0].tolist())

    def get_segment(self):
        """Get one segment of audio. Returns numpy array."""
        if not self.has_mic:
            return None

        # Wait until we have enough samples
        while len(self.buffer) < self.segment_samples and self.is_recording:
            time.sleep(0.1)

        if len(self.buffer) >= self.segment_samples:
            segment = np.array(self.buffer[:self.segment_samples])
            self.buffer = self.buffer[self.segment_samples:]
            return segment
        return None

    def stop(self):
        self.is_recording = False
        if self._stream:
            self._stream.stop()
            self._stream.close()


class ReplayAudioStream:
    """Simulate microphone from a .wav file for testing without hardware."""

    def __init__(self, wav_path, segment_duration=2.0):
        import librosa
        self.y, self.sr = librosa.load(wav_path, sr=16000)
        self.seg_len = int(16000 * segment_duration)
        self.pos = 0

    def get_segment(self):
        if self.pos >= len(self.y):
            return None
        end = min(self.pos + self.seg_len, len(self.y))
        seg = self.y[self.pos:end]
        self.pos = end
        return seg


def run_live(camera_index=0, mic_device=None, replay_audio=None,
             postures=None, room_temp=None, noise_db=None,
             interval=3.0, feed_gap=2.0, sleep_gap=3.0, age_days=42):
    """Run the live multimodal demo."""
    cfg = load_config()
    pipeline = CryAgentPipeline(cfg)
    profile = BabyProfile(age_days=age_days)
    profile.last_feed_time = time.time() - feed_gap * 3600
    profile.last_sleep_end = time.time() - sleep_gap * 3600

    # Init vision
    vision = create_vision_detector()
    camera = None
    if HAS_MEDIAPIPE and replay_audio is None:
        try:
            camera = CameraStream(camera_index=camera_index)
            print(f"[camera] Opened camera {camera_index}")
        except Exception as e:
            print(f"[camera] Cannot open camera: {e}")

    # Init audio
    if replay_audio:
        audio_stream = ReplayAudioStream(replay_audio)
        print(f"[audio] Replay mode: {os.path.basename(replay_audio)}")
    else:
        audio_stream = MicrophoneStream(device=mic_device)
        if audio_stream.has_mic:
            audio_stream.start()
            print(f"[audio] Microphone started (device={mic_device})")
        else:
            print("[audio] No microphone — audio will be skipped")

    print()
    print("=" * 70)
    print("LIVE MULTIMODAL DEMO")
    print("=" * 70)
    print(f"Vision: {'MediaPipe BlazePose' if HAS_MEDIAPIPE else 'annotation fallback'}")
    audio_mode = 'microphone' if hasattr(audio_stream, 'has_mic') and audio_stream.has_mic else ('replay' if replay_audio else 'none')
    print(f"Audio:  {audio_mode}")
    print(f"Context: feed={feed_gap}h ago, sleep={sleep_gap}h ago, age={age_days}d")
    if room_temp:
        print(f"Env:   temp={room_temp}°C, noise={noise_db}dB")
    print(f"Interval: {interval}s")
    print()
    print("Press Ctrl+C to stop")
    print()

    cycle = 0
    try:
        while True:
            cycle += 1
            t0 = time.time()

            # Get vision data
            detected_postures = postures or []
            face_state = None

            if camera is not None:
                ret, frame = camera.read()
                if ret:
                    vision_result = vision.detect_from_frame_full(frame)
                    detected_postures = vision_result["postures"]
                    face_state = vision_result["face_state"]

            # Get audio segment
            audio_seg = audio_stream.get_segment()

            if audio_seg is not None:
                # Write to temp file for pipeline
                import soundfile as sf
                tmp = "/tmp/cry_live_segment.wav"
                sf.write(tmp, audio_seg, 16000)
                audio_path = tmp
            elif replay_audio is None:
                audio_path = None  # No audio available
            else:
                print(f"[{cycle:3d}] Audio replay finished.")
                break

            # Run pipeline
            result = pipeline.run(
                audio_path=audio_path,
                postures=detected_postures,
                face_state=face_state,
                room_temp=room_temp,
                ambient_noise_db=noise_db,
                profile=profile,
            )

            # Print result
            timestamp = time.strftime("%H:%M:%S")
            posture_str = ",".join(detected_postures) if detected_postures else "—"
            face_str = f" face={face_state}" if face_state and face_state != "normal" else ""

            print(f"[{timestamp}] Cycle {cycle:3d}")
            print(f"  Audio:   {result.audio_cause:>12s} ({result.audio_confidence:.0%})")
            print(f"  Vision:  {posture_str}{face_str}")
            print(f"  Fused:   {result.fused_cause:>12s} ({result.fused_confidence:.0%})")
            override = " [OVR]" if result.critical_override else ""
            escalate = " [ESC]" if result.escalated else ""
            print(f"  FINAL:   {result.cause:>12s} ({result.confidence:.0%}) "
                  f"via {result.model}{override}{escalate}")
            print(f"  Latency: {result.latency_ms:.0f}ms")
            print()

            # Clean up temp
            if audio_seg is not None and os.path.exists("/tmp/cry_live_segment.wav"):
                os.unlink("/tmp/cry_live_segment.wav")

            # Wait for next cycle
            elapsed = time.time() - t0
            wait = max(0, interval - elapsed)
            time.sleep(wait)

    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        if camera:
            camera.release()
        if hasattr(audio_stream, 'stop'):
            audio_stream.stop()
        if hasattr(vision, 'close'):
            vision.close()
        print("Stopped.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Live multimodal demo")
    parser.add_argument("--camera", type=int, default=0,
                        help="Camera index (default: 0)")
    parser.add_argument("--mic-device", type=int, default=None,
                        help="Microphone device index")
    parser.add_argument("--replay-audio", type=str, default=None,
                        help="Replay a .wav file instead of microphone")
    parser.add_argument("--postures", type=str, default="",
                        help="Simulated postures (no camera)")
    parser.add_argument("--temp", type=float, default=None)
    parser.add_argument("--noise", type=float, default=None)
    parser.add_argument("--interval", type=float, default=3.0,
                        help="Processing interval in seconds")
    parser.add_argument("--feed-gap", type=float, default=2.0)
    parser.add_argument("--sleep-gap", type=float, default=3.0)
    parser.add_argument("--age", type=int, default=42)
    args = parser.parse_args()

    postures = [p.strip() for p in args.postures.split(",") if p.strip()]

    run_live(
        camera_index=args.camera,
        mic_device=args.mic_device,
        replay_audio=args.replay_audio,
        postures=postures or None,
        room_temp=args.temp,
        noise_db=args.noise,
        interval=args.interval,
        feed_gap=args.feed_gap,
        sleep_gap=args.sleep_gap,
        age_days=args.age,
    )
