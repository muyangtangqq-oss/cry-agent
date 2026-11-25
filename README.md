# Cry-Agent

Agentic baby cry cause diagnosis with multimodal signal fusion,
layered judgment (Tachyone local → cloud judge), and self-learning.

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
                                          ├─ fused ≥ 0.85 → fusion direct
                                          ├─ Tachyone conf ≥ 0.70 → Tachyone
                                          └─ otherwise → cloud judge
                                                     │
                                                     ▼
                                          App notification
                                          + self-learning feedback
```

### Judgment engines

Tachyone and Jev speak the same TypeSafe System One **v1** wire protocol
(`state` + `questions` → typed `answers`), so they are interchangeable:

| Tier | Engine | Measured latency | Role |
|------|--------|------------------|------|
| 1 | Fusion | ~0 ms | multi-source agreement → direct output |
| 2 | Tachyone (local) | 1–12 ms | structured audio/fusion/context scoring |
| 3 | Jev `jev-1.14.0` (cloud) | ~300–600 ms reachable; 1.5 s first 451 retry | genuinely ambiguous cases |

Escalation is tiered: Tachyone answers directly at ≥ 0.70, otherwise Jev is
asked and its answer wins when it comes back valid. Raw confidences are not
compared across engines — they are not calibrated against each other.

```bash
python jev_client.py                    # smoke test: prints model + route
CRY_AGENT_TS_BASE_URL=https://api.typesafe.ai/v1   # skip the Magpie proxy
CRY_AGENT_TS_BASE_URL="https://a/v1,https://b/v1"  # comma-separated failover
CRY_AGENT_TS_MODEL=jev-latest           # alias; jev-preview is also available
CRY_AGENT_TS_MOCK=1                     # force the offline mock for tests
```

TypeSafe returns HTTP `451` from some Cloudflare edges even for a valid key.
The client retries once per route, then uses the clearly-labelled mock
(`model: jev-mock-v0.1`) for 45 s before trying the real API again. The
engine that actually answered is surfaced as `judge_model` in the API and
dashboard, so a mock answer is never mistaken for a real one.

#### Mainland-accessible judge (Jev alternative)

Jev is not reachable from mainland China. `judge_factory.py` can swap the cloud
tier for any OpenAI-compatible model without touching the pipeline:

| Provider | `CRY_AGENT_JUDGE_PROVIDER` | Default model | Key |
|---|---|---|---|
| DeepSeek | `deepseek` | `deepseek-chat` | `CRY_AGENT_JUDGE_API_KEY` or Magpie `deepseek` |
| Qwen (DashScope) | `qwen` | `qwen-plus` | Magpie `qwen` |
| GLM (Zhipu) | `glm` | `glm-4-flash` | Magpie `zhipu` |
| Kimi (Moonshot) | `kimi` | `moonshot-v1-8k` | Magpie `moonshot` |
| Ollama (offline) | `ollama` | `qwen2.5:1.5b-instruct` | none |
| OpenAI | `openai` | `gpt-4.1-mini` | Magpie `openai-api` |

```bash
CRY_AGENT_JUDGE_PROVIDER=deepseek python llm_judge.py   # smoke test
CRY_AGENT_JUDGE_PROVIDER=deepseek python run.py         # use it in the pipeline
CRY_AGENT_JUDGE_PROVIDER=ollama python run.py           # fully offline judge
```

`CRY_AGENT_JUDGE_BASE_URL` and `CRY_AGENT_JUDGE_MODEL` override the preset.
The local Tachyone tier is unchanged and still answers most cases in ≤ 10 ms;
only the ambiguous remainder goes to the cloud model.

These are general instruct models with JSON mode, not calibrated classifiers.
Their answers are marked `calibrated: false` and their confidence is a
self-report, so it must not be compared numerically with Tachyone or Jev
confidences.

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
python test_fusion.py           # confidence calibration: vote share × evidence
python calibrate_fusion_threshold.py  # threshold sweep -> reports/fusion_threshold.md
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

### On Mac mini (headless, launchd)

`deploy_mac.sh` is the one-command install for an Apple Silicon Mac. It
installs Homebrew packages and Python dependencies, clones or pulls the repo,
copies trained models from another Mac, writes `~/.config/cry-agent/env`, and
registers a launchd service that keeps Tachyone + the API running.

```bash
# Copy the script over first if the mini does not have the repo yet
scp deploy_mac.sh mini:'~/deploy_mac.sh'
ssh mini 'chmod +x ~/deploy_mac.sh && ~/deploy_mac.sh --models-from kl@macbook.local'

# Optional: replay/retraining data (~685 MB)
ssh mini '~/deploy_mac.sh --models-from kl@macbook.local \
  --data-from kl@macbook.local:/Users/kl/cry-agent/data'
```

Models are deliberately outside git (`~/.config/cry-agent/models`), so a fresh
mini needs `--models-from`, or `--allow-missing-models` if you plan to train
there. The source Mac must have Remote Login enabled. The script backs up
existing models and is safe to rerun: it pulls the branch, re-syncs models, and
restarts the service.

Common options: `--judge-provider deepseek`, `--port 8000`,
`--no-live-deps`, `--no-launchd`, `--skip-deps`. Run
`./deploy_mac.sh --help` for the full list.

```bash
# Health, logs, restart
curl -fsS http://127.0.0.1:8000/ >/dev/null && echo API_OK
tail -f ~/Library/Logs/cry-agent/api.log
launchctl kickstart -k gui/$(id -u)/com.cryagent.api
```

Camera and microphone permissions must be granted once from a Terminal on the
mini's desktop; launchd cannot show the macOS privacy prompt:

```bash
~/cry-agent/.venv/bin/python ~/cry-agent/live_camera.py --camera 0 --mic-device 0
```

## Components

| File | Purpose |
|------|---------|
| `audio_inference.py` | Cry audio classifier (8 classes, SVM) |
| `multimodal_signals.py` | Signal fusion with critical posture override |
| `scenario_engine.py` | Loads a YAML scenario and builds the pipeline |
| `rules.py` | Numeric / time-window / boolean rule evaluator |
| `configs/*.yaml` | Baby, elderly-care, and pet-behavior scenarios |
| `augment.py` | Pitch / stretch / gain / noise plus channel (EQ/reverb) augmentation |
| `feature_norm.py` | Recording- and owner-level CMVN helpers |
| `train_augmented.py` | Minority-class training + baseline comparison |
| `test_loco.py` | Group CV + leave-one-corpus-out evaluation |
| `test_loco_robust.py` | LOCO comparison of loudness, CMVN, and channel augmentation |
| `ood.py` | Per-class Mahalanobis OOD detector |
| `train_ood.py` | Fit + calibrate OOD gate; LOCO detection report |
| `adapt.py` | CORAL, prior correction, nearest-centroid, family calibrator |
| `test_adaptation.py` | Few-shot LOCO experiment → `reports/adaptation.md` |
| `make_source_stats.py` | Backfill CORAL source moments for an existing model |
| `pilot.py` | Local 3–5 family pilot store, feedback, report, export |
| `vision_mediapipe.py` | MediaPipe BlazePose posture detector |
| `vision_detector.py` | Annotation-based fallback (no camera) |
| `pipeline.py` | End-to-end: fusion → layered judge |
| `judge.py` | Layered judge: audio → Tachyone → cloud judge |
| `judge_factory.py` | Selects Jev or a mainland-accessible LLM judge |
| `llm_judge.py` | OpenAI-compatible judge (DeepSeek/Qwen/GLM/Kimi/Ollama) |
| `tachyone_server.py` | Local judgment server (v1 + legacy request shapes) |
| `tachyone_client.py` | Local client speaking System One v1 |
| `jev_client.py` | Jev cloud client: Magpie → direct, 451 cooldown, labelled mock |
| `audit_dunstan.py` | Dunstan label-semantics and DBL tier audit |
| `typesafe_protocol.py` | Shared v1 request/response helpers and choice rubrics |
| `baby_profile.py` | Personal profile + Bandit self-learning |
| `live_camera.py` | Real-time camera + mic demo |
| `live_replay.py` | File-based streaming simulation |
| `deploy_jetson.sh` | Jetson deployment script |
| `deploy_mac.sh` | Mac mini install: deps, models, launchd service |

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
python pilot.py calibrate --family fam_xxxxxxxx   # needs labelled audio
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
| LOCO + channel aug + CMVN | **52.7% acc / 36.3% balanced** |
| Audio inference latency | ~190 ms warm; ~1.3 s first call |
| Full pipeline latency | ~200–290 ms warm; ~1.1 s first segment |

CV splits are segment-level; segments from the same recording may land in
different folds, so these numbers are an upper bound. See
[`reports/augmentation_cv.md`](reports/augmentation_cv.md) for the full
augmentation comparison and caveats.

**Important: the production model does not generalize across corpora.** In
leave-one-corpus-out evaluation it is worse than the majority baseline on all
three held-out corpora. In-corpus CV and the 97% DBL number are not evidence
of cross-home accuracy. Channel-robust training improves LOCO but costs
in-domain accuracy, so the model is still a within-corpus prototype until
domain adaptation and family pilot data exist. See
[`reports/loco_cv.md`](reports/loco_cv.md).

### Channel / gain robustness

Microphone, room, and gain differences account for much of the LOCO collapse.
Two mitigations are implemented in `augment.py` and `feature_norm.py`:

- **Per-recording CMVN**: subtract each recording's mean feature vector
  before scoring. Recordings with a single segment are left unchanged.
- **Channel augmentation**: smooth 5-band EQ (±6 dB), soft 50–180 Hz and
  3.8–7 kHz band limits, short synthetic reverb, and gain + `tanh` soft clip.
  Augmented copies are added to training folds only.

| Mode | Group CV | Donate | DBL | Dynamic | LOCO mean |
|---|---:|---:|---:|---:|---:|
| base | 76.6 / 41.9 | 79.6 / 22.6 | 20.3 / 19.3 | 21.3 / 36.2 | **40.4 / 26.0** |
| cmvn | 68.2 / 35.4 | 81.0 / 27.8 | 19.9 / 19.4 | 52.4 / 29.0 | 51.1 / 25.4 |
| rms + channel | 77.1 / 42.6 | 77.5 / 27.8 | 20.5 / 19.4 | 25.5 / 40.0 | 41.2 / 29.1 |
| cmvn + channel | 72.2 / 40.5 | 84.3 / 39.0 | 20.3 / 19.3 | 53.5 / 50.5 | **52.7 / 36.3** |

CMVN is the main gain: DynamicSuperb accuracy rises from 21.3% to 53.5% and
Donate balanced accuracy from 22.6% to 39.0%. Channel augmentation adds
roughly 3 pp balanced accuracy on top of CMVN. The trade-off is in-domain:
group CV falls from 76.6% to 72.2%, so CMVN is a cross-device mode, not a
free win for a single known microphone. DBL barely moves because only 52% of
its segments belong to multi-segment recordings, and the rest cannot be
CMVN-normalized.

Live inference keeps a rolling 8-segment buffer per session, so the first
segment runs without CMVN (`cmvn_warmup`). CMVN and CORAL are both
transductive: they use unlabeled audio from the same recording or session.
See [`reports/loco_robust.md`](reports/loco_robust.md) and
`test_loco_robust.py`.

The OOD gate is the containment measure: per-class Mahalanobis distance with
recording-level calibration. On LOCO it flags about 90% of held-out segments
at a 15% false-alarm rate (AUC 0.944), but the 14–16% that pass are still only
16–19% accurate. It also has a session gate: if 8 of the last 12 clips are
OOD, the whole session is marked untrusted. Gated audio is removed from
fusion; vision/context/env can still produce a result, otherwise the output is
`unknown`. See [`reports/ood_gate.md`](reports/ood_gate.md).

### Dunstan labels: what the five sounds mean

InfantCry-DBL labels come from Dunstan's five reflex sounds. The audit in
[`reports/dunstan_audit.md`](reports/dunstan_audit.md) checks our mapping
against the corpus's own definitions:

| DBL | Corpus definition | Mapping | Fit |
|---|---|---|---|
| `neh` | hunger, sucking reflex | hungry | direct |
| `owh` | sleepiness, yawn | sleepy | direct |
| `eh` | upper wind, needs burp | burp | direct |
| `heh` | physical/skin discomfort | discomfort | direct |
| `eairh` | **lower wind / abdominal discomfort** | pain | **too strong** |

`eairh` is lower-wind/gas discomfort, not only sharp pain. We keep it mapped
to `pain` because the `pain` rubric includes colic and DBL's `eairh` segments
are 36% of all pain training data. Re-mapping to `discomfort` gains 5.1 pp
balanced accuracy on DBL alone, but drops LOCO accuracy from 40.4% to 39.3%,
DynamicSuperb pain recall from 33% to 0%, and group-CV pain recall from 41%
to 28%. See [`reports/eairh_mapping.md`](reports/eairh_mapping.md).

The global model still predicts `eairh` as `hungry` 232/241 times, so the
mapping is not the main bottleneck. The bigger caveat is semantic: these are
five *reflex sounds*, not five verified causes. A perfect DBL classifier is
still not a cause diagnosis.

Only the 337 `source_confirmed=TRUE` studio clips were used before. The other
1,214 clips are unconfirmed and 746 have `source=unknown`; adding them raises
Tier-1 test accuracy from 20.3% to 70.5%, but they should be treated as
domain-shift data, not clean ground truth.

## Family Calibration (few-shot)

Per-family calibration is the bridge between the cross-corpus collapse and a
usable pilot. With a small number of parent-confirmed clips, the local model
picks one of four adapters on a recording-level validation slice: global,
CORAL alignment, nearest-centroid prototypes, or CORAL + prototypes.

Leave-one-corpus-out, 30% of recordings used for calibration, 5 seeds:

| Held-out family | Global acc / bal | Calibrated acc / bal | Picked |
|---|---:|---:|---|
| Donate-a-Cry | 79.9% / 23.0% | 71.3% / 47.2% | CORAL |
| InfantCry-DBL | 20.5% / 19.6% | 40.8% / 43.4% | prototype / CORAL+prototype |
| DynamicSuperb | 21.2% / 36.3% | 60.7% / 41.7% | CORAL |
| **Mean** | **40.5% / 26.3%** | **57.6% / 44.1%** | — |

Donate trades 8.6 pp of raw accuracy for +24.2 pp balanced accuracy: the
calibrated model stops predicting `hungry` for almost everything. CORAL uses
unlabeled family audio (mean/covariance only), so raw audio never leaves the
device. Prototypes use only the labelled clips.

```bash
python make_source_stats.py             # backfill CORAL stats for an old model
python pilot.py calibrate --family fam_xxxxxxxx
python pilot.py run --family fam_xxxxxxxx --audio clip.wav   # auto-loads it
python test_adaptation.py --seeds 5     # regenerate reports/adaptation.md
```

For a non-pilot pipeline, point `CRY_AGENT_FAMILY_ADAPTER` at the `.joblib`.
The calibration is a research prototype: labels are parent-reported, the
validation slices are small, and a family-specific bias can be learned. See
[`reports/adaptation.md`](reports/adaptation.md).

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

### Mac mini alternative

A used or existing Apple Silicon Mac mini needs no CUDA and is the fastest way
to stand up a multi-week pilot indoors. It runs the same SVM classifier,
Tachyone server, and dashboard; only the live camera path is slower than a
Jetson. Use [`deploy_mac.sh`](#on-mac-mini-headless-launchd) to install it as
a launchd service, and add a USB microphone plus any UVC webcam.

## Self-Learning

The system improves over time through:
1. **Personal profile** — tracks feed/sleep/diaper intervals
2. **Bandit weight adjustment** — parent feedback adjusts signal weights
3. **Personal cry library** — MFCC embeddings for individual cry matching
4. **Misjudgment correction** — errors fed to Jev as context for future judgments
5. **Pilot feedback store** — real-family outcomes for later offline retraining
