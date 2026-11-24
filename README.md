# Cry-Agent

Agentic baby cry cause diagnosis with multimodal signal fusion,
layered judgment (Tachyone local → Jev cloud), and self-learning.

## Architecture

```
Sensors                    Processing              Output
─────────                  ──────────              ──────
Microphone → Audio Classifier (SVM, 8 classes)     ┐
Camera → MediaPipe BlazePose (posture)             ├→ Multimodal
Time → Context (feed/sleep/diaper)                 │   Fuser
DHT22 → Temperature                                 │
USB → Noise level                                   ┘
                                                     │
                                                     ▼
                                          Layered Judge
                                          ├─ conf ≥ 0.80 → direct
                                          ├─ 0.40-0.80 → Tachyone
                                          └─ < 0.40 → Jev
                                                     │
                                                     ▼
                                          App notification
                                          + self-learning feedback
```

## Quick Start

### On dev machine (no hardware needed)

```bash
# Setup
uv venv --python 3.12
source .venv/bin/activate
pip install librosa soundfile scikit-learn joblib imbalanced-learn httpx

# Train (downloads Donate-a-Cry + uses InfantCry-DBL + DynamicSuperb)
python train_combined_v2.py

# Test without hardware
python tachyone_server.py 8901 &
python test_full_pipeline.py    # 6 scenarios, real audio
python live_replay.py <wav> --postures rooting  # streaming simulation
```

### On Jetson Orin Nano (with camera + mic)

```bash
# Deploy
./deploy_jetson.sh

# Run live
python live_camera.py --camera 0 --mic-device 0 --temp 22 --interval 3

# Or replay mode (no camera needed)
python live_camera.py --replay-audio <wav> --postures rooting
```

## Components

| File | Purpose |
|------|---------|
| `audio_inference.py` | Cry audio classifier (8 classes, SVM) |
| `multimodal_signals.py` | Signal fusion with critical posture override |
| `vision_mediapipe.py` | MediaPipe BlazePose posture detector |
| `vision_detector.py` | Annotation-based fallback (no camera) |
| `pipeline.py` | End-to-end: fusion → layered judge |
| `judge.py` | Layered judge: audio → Tachyone → Jev |
| `tachyone_server.py` | Local judgment server |
| `jev_client.py` | Jev cloud client (with mock fallback) |
| `baby_profile.py` | Personal profile + Bandit self-learning |
| `live_camera.py` | Real-time camera + mic demo |
| `live_replay.py` | File-based streaming simulation |
| `deploy_jetson.sh` | Jetson deployment script |

## Model Performance

| Metric | Value |
|--------|-------|
| Training data | Donate-a-Cry + InfantCry-DBL + DynamicSuperb |
| Classes | 8 (hungry/sleepy/discomfort/pain/burp/lonely/startle/temperature) |
| CV Accuracy | 79.6% |
| CV Balanced Accuracy | 44.7% |
| Cross-dataset accuracy | 96.1% (on InfantCry-DBL) |
| Per-class accuracy | 82-100% (on DynamicSuperb) |
| Audio inference latency | ~200ms |
| Full pipeline latency | ~600ms |

## Hardware (Jetson deployment)

| Part | Model | Price |
|------|-------|-------|
| Board | Jetson Orin Nano 8GB | ¥1500 |
| Storage | 128GB NVMe SSD | ¥80 |
| Camera | Raspberry Pi Camera V3 | ¥120 |
| Microphone | USB microphone | ¥40 |
| Temp sensor | DHT22 | ¥15 |
| Noise sensor | USB decibel module | ¥40 |
| Total | | ~¥1900 |

## Self-Learning

The system improves over time through:
1. **Personal profile** — tracks feed/sleep/diaper intervals
2. **Bandit weight adjustment** — parent feedback adjusts signal weights
3. **Personal cry library** — MFCC embeddings for individual cry matching
4. **Misjudgment correction** — errors fed to Jev as context for future judgments
