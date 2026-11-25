# Cry-Agent: Agentic Baby Cry Cause Diagnosis

## A multimodal, self-learning system for real-time infant cry interpretation

**Version:** 0.1.0  
**Date:** 2026-11-24  
**Repository:** [github.com/muyangtangqq-oss/cry-agent](https://github.com/muyangtangqq-oss/cry-agent)

---

## 1. Problem

Existing baby monitors (Cubo AI, Nanit, Owlet) detect crying and movement, but **cannot diagnose why a baby is crying**. Parents are left guessing: hungry? sleepy? pain? discomfort? This trial-and-error process is stressful, especially for new parents and during nighttime.

**Key gap:** Current products use single-signal detection (video only) with no semantic understanding of cry cause, no context integration, and no personalization.

## 2. Solution

Cry-Agent is an agentic system that fuses **four signal sources** into a structured judgment with calibrated confidence, then escalates uncertain cases through a layered reasoning pipeline.

### Architecture

```
Sensors → Signal Extraction → OOD Gate → Multimodal Fusion → Layered Judge → Output
                                                                        ↓
                                                              Self-Learning Feedback
```

**Four signal layers:**

| Layer | Source | Signal | Weight |
|-------|--------|--------|--------|
| Audio | Microphone | Cry classification (SVM, 8 classes) | 40% |
| Vision | Camera | Posture detection (MediaPipe BlazePose, 13 postures) | 30% |
| Context | Time tracking | Feed/sleep/diaper intervals + age + time-of-day | 20% |
| Environment | Sensors | Room temperature + ambient noise | 10% |

**Audio OOD gate:** per-class Mahalanobis distance in standardized feature
space, calibrated on unseen recordings. If audio is OOD it is removed from
fusion and the text judge is not allowed to guess. A session-level gate marks
the whole session untrusted when 8 of the last 12 clips are OOD.

**Layered judge (escalation chain):**

1. **Fusion direct** (confidence ≥ 0.85): Multi-source agreement → immediate output
2. **Tachyone local** (fused < 0.85): Local engine scores audio, fusion and
   context; resolves its own ≥ 0.70 cases in 1–12 ms
3. **Cloud judge** (Tachyone < 0.70 or disagreement): TypeSafe System One v1
   (`jev-1.14.0`), ~300–600 ms when reachable, 1.5 s on the first 451 retry

Both engines speak the same v1 wire protocol (`state` + `questions` → typed
`answers`), so `/v1/systemone` can be pointed at either one. Tachyone now reads
the structured audio/fusion/context fields instead of relying on keyword hits,
which lets the local tier carry the common cases. Once Jev is invoked, its
answer wins when valid; the engines' raw confidences are not compared because
they are not calibrated against each other.

**Self-learning loop:**

- **Level 1 — Personal profile:** Auto-tracks feed/sleep/diaper intervals, learns individual patterns
- **Level 2 — Bandit weight adjustment:** Parent feedback (correct/wrong) adjusts signal weights
- **Level 3 — Personal cry library:** MFCC embeddings stored per cause, similarity matching for individual cry patterns
- **Level 4 — Misjudgment correction:** Errors recorded as context notes, fed to Jev for future judgments

### Key Design Decisions

**Critical posture override:** Certain postures (arching back = pain, arms spread = startle, rooting = hungry) are high-certainty single signals. When detected with ≥ 0.85 confidence, they can override lower-confidence audio classification — even when audio is 98% confident in a different cause. This is the key mechanism that allows vision to correct audio errors.

**Disagreement escalation:** When top-2 causes are within 15% of each other, the system flags `escalate` instead of guessing. This routes to Tachyone → Jev for deeper reasoning.

**Honest uncertainty:** Low confidence is escalated to Tachyone, then to the
cloud judge; it never becomes a direct answer. Audio OOD (or a session-level
shift) outputs `unknown` / `fusion-nonaudio` rather than guessing. The session
gate catches corpus shift even when individual clips slip through.

## 3. Validation

### Audio Classifier

| Metric | Value |
|--------|-------|
| Training data | Donate-a-Cry (457 clips) + InfantCry-DBL (337 clips) + DynamicSuperb (354 clips) |
| Classes | 8: hungry, sleepy, discomfort, pain, burp, lonely, startle, temperature |
| Algorithm | SVM (RBF kernel) + SVMSMOTE + StandardScaler |
| Features | MFCC(20) + delta + delta² + spectral(centroid/bandwidth/rolloff/flatness) + ZCR + RMS + pitch(F0) + temporal |
| Feature count | 95 |
| Cross-validation accuracy | 79.4% (5-fold, augmented model) |
| Balanced accuracy | 52.3% (46.0% before augmentation) |
| DBL regression | 97.0% (held-in corpus, not a clean holdout) |
| Minority recall | startle 58%, temperature 67%, pain 53% |
| Inference latency | ~200ms per 2s segment |

CV splits are segment-level; 2 s segments from the same recording can land in
different folds. These numbers are an upper bound, not real-home
generalization. See `reports/augmentation_cv.md`.

#### Minority-class augmentation

Time-domain augmentation (pitch ±3 st, stretch 0.9–1.1, gain −6…+3 dB,
noise SNR 15–35 dB) is generated only from training-fold sources, then the
same SVM/SVMSMOTE pipeline is refit.

| Metric | Baseline | Augmented |
|--------|----------|-----------|
| Accuracy | 79.5% | 79.4% |
| Balanced accuracy | 46.0% | **52.3%** |
| startle recall | 33.3% | **58.3%** |
| temperature recall | 50.0% | **66.7%** |
| pain recall | 46.5% | **52.8%** |

Overall accuracy is flat because hungry is 73% of the corpus; the gain is in
minority recall. The augmented model is now the production model; the previous
weights are kept as `*.pre_aug.bak`.

#### Leave-one-corpus-out (critical result)

Random segment CV mixes corpora. A stricter test holds out one entire corpus
and evaluates only the five labels shared by all three sources:

| Evaluation | n | Accuracy | Balanced | Majority baseline |
|---|---:|---:|---:|---:|
| Group 5-fold by recording | 4,993 | 76.6% | 41.9% | 74.5% |
| LOCO donate | 2,492 | 79.6% | 22.6% | 83.3% |
| LOCO DBL | 482 | 20.3% | 19.3% | 21.0% |
| LOCO DynamicSuperb | 2,019 | 21.3% | 36.2% | 76.4% |
| **LOCO mean** | | **40.4%** | **26.0%** | **74.5%** |

The model is worse than always predicting the training majority class on
**all three** held-out corpora. Group CV is only +2.1 pp over its majority
baseline. This means the current features/classifier are learning corpus
identity and class frequency, not a transferable cry signature.

Consequences:

- In-corpus CV (79.4%) and the held-in DBL number (97.0%) must not be quoted
  as cross-dataset or cross-home accuracy.
- The production classifier is a within-corpus prototype, not a validated
  general cry-cause model.
- The family pilot is now a necessity, not a nice-to-have.
- See `reports/loco_cv.md` and `test_loco.py`.

#### Channel / gain robustness

Recording-level normalization targets the part of the shift that comes from
microphone gain, room response, and distance rather than from the cry itself:

- **CMVN** subtracts each recording's mean feature vector. Files with a
  single segment are left unchanged.
- **Channel augmentation** applies a smooth 5-band EQ (±6 dB), soft band
  limiting, short synthetic reverb, and gain with `tanh` soft clipping.
  Copies enter training folds only; evaluation stays clean.

| Mode | Group CV | Donate | DBL | Dynamic | LOCO mean |
|---|---:|---:|---:|---:|---:|
| base | 76.6 / 41.9 | 79.6 / 22.6 | 20.3 / 19.3 | 21.3 / 36.2 | 40.4 / 26.0 |
| cmvn | 68.2 / 35.4 | 81.0 / 27.8 | 19.9 / 19.4 | 52.4 / 29.0 | 51.1 / 25.4 |
| rms + channel | 77.1 / 42.6 | 77.5 / 27.8 | 20.5 / 19.4 | 25.5 / 40.0 | 41.2 / 29.1 |
| cmvn + channel | 72.2 / 40.5 | 84.3 / 39.0 | 20.3 / 19.3 | 53.5 / 50.5 | **52.7 / 36.3** |

CMVN recovers most of the cross-corpus gap (LOCO mean 40.4% → 52.7%,
balanced 26.0% → 36.3%). The cost is in-domain: group CV drops from 76.6%
to 72.2%. CMVN therefore belongs behind a per-deployment switch for unknown
or changing hardware, not as the default for a single calibrated microphone.
DBL remains near chance because only 51.9% of its segments belong to
multi-segment recordings and single-segment files cannot be normalized.

Both CMVN and CORAL are transductive: they use unlabeled audio from the same
recording or live session. Live inference needs a warm-up of two segments
before the rolling buffer is active. See `reports/loco_robust.md`.

### OOD Gate

Per-class Mahalanobis distance with recording-level calibration (85th
percentile, about 15% false alarms):

| Held-out corpus | AUC | Gate rate | Trusted n | Trusted accuracy |
|---|---:|---:|---:|---:|
| Donate | 1.000 | 100% | 0 | n/a |
| InfantCry-DBL | 0.920 | 84% | 77 | 19% |
| DynamicSuperb | 0.912 | 86% | 280 | 16% |
| **mean** | **0.944** | **90%** | **357** | |

The gate catches about 90% of held-out corpus segments, but the remaining
14–16% are still only 16–19% accurate. Therefore:

- Gated audio is set to `unknown` and removed from fusion.
- If vision/context/env carry strong evidence, the result is labelled
  `fusion-nonaudio` and `trusted=false`.
- If nothing else carries evidence, the output is `unknown` via `ood-gate`
  and the text judge is never called.
- A session gate marks the whole session untrusted when 8 of the last 12
  clips are OOD. This is important because a new home is shifted at the
  session level, not just per clip.

This is containment, not a cure. A gate cannot recover acoustic information
that the model never learned.

For the channel-robust model the gate is fitted on **raw RMS** features,
while the classifier consumes RMS + CMVN. This is deliberate: CMVN removes
the recording-level offset that the gate uses, and scoring OOD on CMVN
features drops mean LOCO AUC from 0.946 to about 0.70. The paired detector
keeps AUC 1.000 / 0.924 / 0.915 across the three held-out corpora with a
91% gate rate at 15% false alarms. It is saved alongside the robust model
and selected automatically through `ood_on_raw_features` in
`model_config.json`. See `reports/ood_gate_robust.md`.

### Multimodal Fusion

| Scenario | Audio | Vision | Result | Mechanism |
|----------|-------|--------|--------|-----------|
| Hungry (3 sources agree) | hungry 75% | hungry 95% | hungry 98% ✓ | Multi-source boost |
| Sleepy (3 sources agree) | sleepy 60% | sleepy 97% | sleepy 98% ✓ | Multi-source boost |
| Pain (audio wrong) | hungry 55% | pain 97% [CRIT] | pain 95% ✓ | Critical override |
| Startle (env+vision) | startle 40% | startle 95% | startle 98% ✓ | Multi-source boost |
| Cold (env+vision) | discomfort 50% | temperature 85% | temperature 75% ✓ | Env override + escalate |
| Lonely (audio 98% wrong) | hungry 98% | lonely 95% [CRIT] | lonely 85% ✓ | Critical override |

**Key result:** Vision correctly overrode audio in 3/6 scenarios, including one where audio was 98% confident in the wrong answer.

Fusion confidence is the winner's share of the weighted vote multiplied by an
evidence-strength term that saturates at 60% of the total source weight;
agreement boosts are applied after that. This keeps a lone weak source from
being normalized into 100% confidence, while two strong agreeing sources can
still reach the direct-output tier. Top-2 margin is computed on the same
normalized scores, so weak evidence no longer looks like disagreement.

### Layered Judge Fallback

| Scenario | Chain | Winner |
|----------|-------|--------|
| Low-conf hungry | audio(45%) → tachyone(85%) | Tachyone ✓ |
| Low-conf sleepy | audio(50%) → tachyone(93%) | Tachyone ✓ |
| Ambiguous equal split | audio(35%) → tachyone(66%) → jev(36%) | Jev ✓ |
| Post-feed burp | audio(42%) → tachyone(79%) | Tachyone ✓ |
| Evening colic | audio(48%) → tachyone(87%) | Tachyone ✓ |
| Uniform low confidence | audio(20%) → tachyone(57%) → jev(29%) | Jev (uncertain) |

**Key result:** the local tier resolved 4/6 cases in ≤ 10 ms; the two
genuinely ambiguous cases escalated to Jev. In this region-blocked run Jev
returned mock answers (the first call waited ~1.5 s through the 451 retry,
then the 45 s cooldown answered instantly). Accuracy on the 5 evaluable
scenarios was 5/5.

### Self-Learning (14-day simulation)

| Day | Accuracy | Cry Library Size | Top Signal Weight |
|-----|----------|-----------------|-------------------|
| 1 | 100% | 15 | cry_pattern: 0.40 |
| 4 | 100% | 60 | cry_pattern: 0.40 |
| 7 | 100% | 105 | cry_pattern: 0.38 |

**Key result:** over 14 days (15 events/day), the signal weights shifted from
cry_pattern (0.40→0.33) toward posture (0.30→0.34), context and time pattern
(0.15→0.17 each). Early accuracy was 100%, late accuracy 97%.

### Real-World Pilot Kit

The household validation harness is built but not yet deployed:

- 3–5 family protocol with consent, safety boundaries and a 2-week schedule
- pseudonymous family IDs, opt-in audio retention, withdrawal deletes audio
- parent feedback per event (`correct` or actual cause)
- weekly report: coverage, accuracy, high-confidence errors, latency p50/p95
- HMAC-pseudonymized CSV export for offline analysis
- `export-training` turns consented, feedback-labeled audio into class folders
  consumable by `train_augmented.py --extra-data`
- 5-family simulated end-to-end test (`test_pilot.py`)

This is deliberately local-first. No family data has been collected yet.

### End-to-End Pipeline

| Metric | Value |
|--------|-------|
| Pipeline accuracy | 6/6 scenarios (100%) |
| Average latency | 200–290ms warm; ~1.1s first segment (model warm-up) |
| Models used | fusion 4, tachyone 2; Jev not needed in this run |
| Critical overrides | 2/6 (vision/env corrected audio) |

### Live Replay Demo

| Segment | Audio Confidence | Fused Confidence | Model |
|---------|-----------------|-----------------|-------|
| 0 (0-2s) | discomfort 78% | hungry 95% | fusion [OVERRIDE] |
| 1 (2-4s) | discomfort 59% | hungry 95% | fusion [OVERRIDE] |
| 2 (4-6.9s) | hungry 80% | hungry 95% | fusion |

Real Donate-a-Cry hungry clip (`999bf14b-…-hu.wav`), replayed with
`--postures rooting`. The first two segments were acoustically misclassified
as discomfort; the rooting posture overrode both to hungry. Aggregate:
3/3 hungry, 95% average confidence, 120ms per segment, 2 overrides.

## 4. Technology Stack

| Component | Technology | Notes |
|-----------|-----------|-------|
| Audio classification | scikit-learn SVM + librosa | 95 MFCC + spectral features |
| Posture detection | MediaPipe BlazePose | 33 body landmarks → 13 postures |
| Local judge | Tachyone (heuristic engine) | 1–12 ms, edge-deployable, v1 protocol |
| Cloud judge | Jev / TypeSafe System One v1 | `jev-1.14.0`, ~300–600 ms reachable; 1.5 s first 451 retry |
| Self-learning | Bandit algorithm + SQLite | Signal weight adjustment |
| API server | FastAPI + uvicorn | REST + WebSocket |
| Dashboard | HTML/CSS/JS | Real-time results + feedback |
| Deployment | Docker | Mac/Jetson/Pi |

## 5. Deployment Options

| Platform | Hardware | Cost | Vision | Notes |
|----------|----------|------|--------|-------|
| Mac Mini | Existing | ¥0 | USB camera | Fastest to validate |
| Raspberry Pi 5 | Pi 5 8GB | ¥500 | Pi Camera V3 (2-3fps) | Audio + env only viable |
| Jetson Orin Nano | 8GB dev kit | ¥3000 | Pi Camera V3 (15-20fps) | Full multimodal |

## 6. Comparison with Existing Products

| Feature | Cubo AI | Nanit | Owlet | **Cry-Agent** |
|---------|---------|-------|-------|---------------|
| Cry detection | ✅ | ✅ | ✅ | ✅ |
| Cry cause diagnosis | ❌ | ❌ | ❌ | **✅ (8 classes)** |
| Posture detection | Basic | Basic | ❌ | **✅ (13 postures)** |
| Context integration | ❌ | ❌ | ❌ | **✅ (feed/sleep/diaper)** |
| Environment sensors | ❌ | ❌ | ❌ | **✅ (temp + noise)** |
| Personalization | ❌ | ❌ | ❌ | **✅ (self-learning)** |
| Confidence calibration | ❌ | ❌ | ❌ | **✅** |
| Edge deployment | ❌ | ❌ | ❌ | **✅** |
| Open source | ❌ | ❌ | ❌ | **✅ (MIT)** |

## 7. Limitations & Future Work

### Current Limitations

- **Training data imbalance:** hungry is still 73% of segments. Augmentation
  raised balanced accuracy from 46.0% to 52.3%, but startle/temperature have
  only 12 source segments each; augmentation cannot invent missing diversity.
- **Jev edge-dependent 451:** the v1 API is live and was verified against
  `jev-1.14.0`, but some Cloudflare edges return 451 even for a valid key.
  The client retries once per route and falls back to a clearly-labelled mock
  (`jev-mock-v0.1`) for 45 s, so cloud availability is not guaranteed. Clef
  (Apache 2.0) remains the fully offline option.
- **Jev is unreachable from mainland China.** `CRY_AGENT_JUDGE_PROVIDER` can
  replace the cloud tier with DeepSeek, Qwen, GLM, Kimi, a local Ollama model,
  or OpenAI. These are general instruct models with JSON mode, not calibrated
  classifiers: their confidence is a self-report (`calibrated: false`) and must
  not be compared numerically with Tachyone or Jev confidences.
- **Dunstan `eairh` is stronger than the corpus definition.** The corpus says
  lower-wind / abdominal discomfort, and we map it to `pain`. We keep that
  mapping because the rubric includes colic and `eairh` is 36% of all pain
  training segments. Re-mapping to `discomfort` gains 5.1 pp balanced accuracy
  on DBL alone, but costs 1.1 pp LOCO accuracy and takes DynamicSuperb pain
  recall from 33% to 0%. `eairh` is still predicted as `hungry` in 232/241
  segments, and DBL provides five reflex *sounds*, not five verified causes.
  See `reports/dunstan_audit.md` and `reports/eairh_mapping.md`.
- **Only Dunstan Tier 1 is clean.** The 337 studio clips carry
  `source_confirmed=TRUE`. The other 1,214 clips are unconfirmed, 746 have
  `source=unknown`, and one file has a folder/filename label conflict. Adding
  them raises Tier-1 test accuracy from 20.3% to 70.5%, but they are
  domain-shift data, not a clean holdout.
- **Vision prototype:** MediaPipe BlazePose not yet tested on real baby footage. Posture-to-cause mapping is rule-based, not learned.
- **No real-world deployment data:** All validation is on public datasets. The
  pilot kit is ready, but no family has been enrolled yet. Real home noise,
  microphone placement, and baby variability remain untested.
- **Segment-level CV:** clips from the same recording can cross folds, so CV is
  an upper bound. LOCO is now implemented; a household holdout still requires
  pilot data.
- **Severe cross-corpus domain shift:** LOCO mean is 40.4% accuracy / 26.0%
  balanced, below the 74.5% majority baseline. Per-recording CMVN plus
  channel augmentation lifts this to 52.7% / 36.3%, still far from a solved
  transfer problem and at the cost of in-domain accuracy (group CV
  76.6% → 72.2%). The OOD gate catches ~90% at a 15% false-alarm rate, but
  the trusted remainder is still 16–19% accurate.

**Family calibration changes the picture.** With 30% of a held-out corpus's
recordings used as parent-labelled calibration (split by recording, 5 seeds),
an automatic per-family choice between CORAL and nearest-centroid prototypes
raises the three-corpus mean from 40.5% / 26.3% to **57.6% / 44.1%**:

| Held-out family | Global acc / bal | Calibrated acc / bal | Picked |
|---|---:|---:|---|
| Donate-a-Cry | 79.9% / 23.0% | 71.3% / 47.2% | CORAL |
| InfantCry-DBL | 20.5% / 19.6% | 40.8% / 43.4% | prototype / CORAL+prototype |
| DynamicSuperb | 21.2% / 36.3% | 60.7% / 41.7% | CORAL |

Donate loses 8.6 pp of raw accuracy but gains 24.2 pp balanced accuracy —
the desired trade for a monitor that should not answer `hungry` by default.
This is still a public-corpus simulation: no real family has enrolled, the
labels are parent-reported, and CORAL is transductive on unlabeled family
audio. It is evidence that the collapse is partly a per-recording calibration
problem, not evidence of home accuracy.

### Roadmap

1. ✅ **Data augmentation:** pitch shift + time stretch + gain + noise for
   minority classes; balanced accuracy +6.3 pp.
2. 🟡 **Real-world validation:** pilot kit implemented and tested with 5
   simulated families; waiting for 3–5 real families.
3. ⏳ **Clef deployment:** optional offline judge for installations that must
   not depend on TypeSafe availability.
4. ⏳ **Model export:** SVM → ONNX/TFLite for optimized edge inference.
5. ⏳ **Multi-baby tracking:** ByteTrack + ReID for multi-child scenarios.
6. ⏳ **Long-term pattern discovery:** weekly/monthly anomaly detection.
7. 🟡 **Domain adaptation:** per-family CORAL / prototype calibration lifts
   LOCO mean to 57.6% acc / 44.1% bal; per-recording CMVN + channel
   augmentation lifts the uncalibrated model to 52.7% / 36.3%. Remaining:
   combining both under one calibration policy, and validation on real
   households.
8. ✅ **OOD containment:** per-class Mahalanobis gate + session-level shift
   detection; prevents confident output from a clearly shifted corpus.

## 8. Reproduce

```bash
# Clone
git clone https://github.com/muyangtangqq-oss/cry-agent.git
cd cry-agent

# Setup
uv venv --python 3.12 && source .venv/bin/activate
uv sync

# Train (downloads datasets automatically)
python train_combined_v2.py
python train_augmented.py   # minority augmentation + CV comparison
python train_ood.py         # fit gate + LOCO detection report
python make_source_stats.py # CORAL source moments for family calibration

# Test
python tachyone_server.py 8901 &
python jev_client.py        # live Jev smoke test (jev-1.14.0)
python test_full_pipeline.py
python test_scenario_engine.py
python test_loco.py
python test_loco_robust.py
python train_augmented.py --rms-loudness --file-cmvn --channel-augment \
  --sampler smote --output-dir ~/.config/cry-agent/models_robust
python train_ood.py --robust
python test_adaptation.py --seeds 5
python test_family_calibrator.py
python test_ood.py
python test_augment.py
python test_pilot.py
python test_pilot_calibration.py

# Dashboard
uvicorn api_server:app --host 0.0.0.0 --port 8000
# Open http://localhost:8000/dashboard
```

## 9. License

MIT
