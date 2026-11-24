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
pip install librosa soundfile scikit-learn joblib imbalanced-learn httpx pyyaml

# Train (downloads Donate-a-Cry + uses InfantCry-DBL + DynamicSuperb)
python train_combined_v2.py

# Optional: minority-class augmentation comparison (writes models_aug/)
python train_augmented.py
# After a pilot: include household clips exported by pilot.py
python train_augmented.py --extra-data pilot_training_set

# Optional: fit the out-of-distribution gate
python train_ood.py

# Test without hardware
python tachyone_server.py 8901 &
python test_full_pipeline.py    # 6 scenarios, real audio
python test_ood.py              # gate: normal vs noise vs session shift
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
| `scenario_engine.py` | Loads a YAML scenario and builds the pipeline |
| `rules.py` | Numeric / time-window / boolean rule evaluator |
| `configs/*.yaml` | Baby, elderly-care, and pet-behavior scenarios |
| `augment.py` | Pitch / stretch / gain / noise augmentation |
| `train_augmented.py` | Minority-class training + baseline comparison |
| `test_loco.py` | Group CV + leave-one-corpus-out evaluation |
| `ood.py` | Per-class Mahalanobis OOD detector |
| `train_ood.py` | Fit + calibrate OOD gate; LOCO detection report |
| `pilot.py` | Local 3–5 family pilot store, feedback, report, export |
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

## Multi-Scenario (config-driven)

The same inference stack powers different monitoring domains. A scenario YAML
declares the audio classes, posture→cause map, context/env rules, fusion
weights, override policy, judge thresholds, and profile defaults:

```yaml
signals:
  audio:
    classes: [fall_sound, groan, call_for_help, coughing, normal_speech, silence]
  vision:
    posture_correlation:
      lying_down: {fall: 0.90, unconscious: 0.70}
  env:
    sensors:
      floor_pressure:
        rules:
          - condition: "> 50"
            causes: {fall: 0.85}
fusion:
  weights: {audio: 0.30, vision: 0.35, context: 0.25, env: 0.10}
  critical_override:
    min_vision_confidence: 0.80
  env_override:
    causes: [fall, gas_leak]
```

Included scenarios: `baby_cry`, `elderly_care`, `pet_behavior`.

```bash
python main.py --list
python main.py --describe elderly_care
python main.py --scenario elderly_care --postures lying_down \
  --sensor floor_pressure=60 --json
python test_scenario_engine.py   # baby regression + elderly/pet extension
CRY_AGENT_SCENARIO=elderly_care python live_camera.py --camera 0

python test_loco.py             # group CV + leave-one-corpus-out report
```

Adding a domain means adding a YAML file; training its audio classifier is the
only domain-specific model work.

## Real-World Pilot

The pilot kit is local-first and ready to deploy to 3–5 volunteer families:

```bash
python pilot.py init --consent --age-days 42 --audio-retention
python pilot.py run --family fam_xxxxxxxx --audio clip.wav \
  --postures rooting --temp 22 --feed-gap 3.0
python pilot.py feedback --event 12 --cause hungry
python pilot.py report
python pilot.py export --out pilot_export_week1
python pilot.py export-training --out pilot_training_set
```

Audio retention is opt-in, family IDs are pseudonymous, exports are
HMAC-pseudonymized, and `withdraw` deletes that family's audio. The full
consent, safety, feedback and analysis protocol is in
[`PILOT_PROTOCOL.md`](PILOT_PROTOCOL.md).

No families have been enrolled yet; `test_pilot.py` exercises the full loop
with 5 simulated families.

## Model Performance

| Metric | Value |
|--------|-------|
| Training data | Donate-a-Cry + InfantCry-DBL + DynamicSuperb |
| Training segments | 5,053 (2 s, 16 kHz) |
| Classes | 8 (hungry/sleepy/discomfort/pain/burp/lonely/startle/temperature) |
| CV Accuracy | 79.4% |
| CV Balanced Accuracy | 52.3% (46.0% before augmentation) |
| Minority recall | startle 58%, temperature 67%, pain 53% |
| DBL regression | 97.0% (held-in corpus, not a clean holdout) |
| Group 5-fold by recording | 76.6% acc / 41.9% balanced (majority 74.5%) |
| Leave-one-corpus-out mean | **40.4% acc / 26.0% balanced** (majority 74.5%) |
| Audio inference latency | ~200ms |
| Full pipeline latency | ~600ms |

CV splits are segment-level; segments from the same recording may land in
different folds, so these numbers are an upper bound. See
[`reports/augmentation_cv.md`](reports/augmentation_cv.md) for the full
augmentation comparison and caveats.

**Important: the current model does not generalize across corpora.** In
leave-one-corpus-out evaluation it is worse than the majority baseline on all
three held-out corpora. In-corpus CV and the 97% DBL number are not evidence
of cross-home accuracy. The model should be treated as a within-corpus
prototype until domain adaptation and family pilot data exist. See
[`reports/loco_cv.md`](reports/loco_cv.md).

The OOD gate is the containment measure: per-class Mahalanobis distance with
recording-level calibration. On LOCO it flags about 90% of held-out segments
at a 15% false-alarm rate (AUC 0.944), but the 14–16% that pass are still only
16–19% accurate. It also has a session gate: if 8 of the last 12 clips are
OOD, the whole session is marked untrusted. Gated audio is removed from
fusion; vision/context/env can still produce a result, otherwise the output is
`unknown`. See [`reports/ood_gate.md`](reports/ood_gate.md).

## Hardware (Jetson deployment)

| Part | Model | Price |
|------|-------|-------|
| Board | Jetson Orin Nano 8GB developer kit | ¥3000 |
| Storage | 128GB NVMe SSD | ¥80 |
| Camera | Raspberry Pi Camera V3 | ¥120 |
| Microphone | USB microphone | ¥40 |
| Temp sensor | DHT22 | ¥15 |
| Noise sensor | USB decibel module | ¥40 |
| Total | | ~¥3400 |

## Self-Learning

The system improves over time through:
1. **Personal profile** — tracks feed/sleep/diaper intervals
2. **Bandit weight adjustment** — parent feedback adjusts signal weights
3. **Personal cry library** — MFCC embeddings for individual cry matching
4. **Misjudgment correction** — errors fed to Jev as context for future judgments
5. **Pilot feedback store** — real-family outcomes for later offline retraining
