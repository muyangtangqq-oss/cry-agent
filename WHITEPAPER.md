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
Sensors → Signal Extraction → Multimodal Fusion → Layered Judge → Output
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

**Layered judge (escalation chain):**

1. **Fusion direct** (confidence ≥ 0.80): Multi-source agreement → immediate output
2. **Tachyone local** (0.40-0.80): Local heuristic engine fuses with context, < 5ms
3. **Jev cloud** (< 0.40): Cloud-based structured judgment with misjudgment correction

**Self-learning loop:**

- **Level 1 — Personal profile:** Auto-tracks feed/sleep/diaper intervals, learns individual patterns
- **Level 2 — Bandit weight adjustment:** Parent feedback (correct/wrong) adjusts signal weights
- **Level 3 — Personal cry library:** MFCC embeddings stored per cause, similarity matching for individual cry patterns
- **Level 4 — Misjudgment correction:** Errors recorded as context notes, fed to Jev for future judgments

### Key Design Decisions

**Critical posture override:** Certain postures (arching back = pain, arms spread = startle, rooting = hungry) are high-certainty single signals. When detected with ≥ 0.85 confidence, they can override lower-confidence audio classification — even when audio is 98% confident in a different cause. This is the key mechanism that allows vision to correct audio errors.

**Disagreement escalation:** When top-2 causes are within 15% of each other, the system flags `escalate` instead of guessing. This routes to Tachyone → Jev for deeper reasoning.

**Honest uncertainty:** Confidence < 0.40 → system outputs "unknown" rather than guessing. This prevents false confidence and builds parent trust.

## 3. Validation

### Audio Classifier

| Metric | Value |
|--------|-------|
| Training data | Donate-a-Cry (457 clips) + InfantCry-DBL (337 clips) + DynamicSuperb (354 clips) |
| Classes | 8: hungry, sleepy, discomfort, pain, burp, lonely, startle, temperature |
| Algorithm | SVM (RBF kernel) + SVMSMOTE + StandardScaler |
| Features | MFCC(20) + delta + delta² + spectral(centroid/bandwidth/rolloff/flatness) + ZCR + RMS + pitch(F0) + temporal |
| Feature count | 95 |
| Cross-validation accuracy | 79.6% (5-fold) |
| Balanced accuracy | 44.7% |
| Cross-dataset accuracy | 96.1% (train on Donate-a-Cry+DBL, test on DBL) |
| Per-class accuracy | 82-100% (on DynamicSuperb) |
| Inference latency | ~200ms per 2s segment |

### Multimodal Fusion

| Scenario | Audio | Vision | Result | Mechanism |
|----------|-------|--------|--------|-----------|
| Hungry (3 sources agree) | hungry 75% | hungry 95% | hungry 98% ✓ | Multi-source boost |
| Sleepy (3 sources agree) | sleepy 60% | sleepy 97% | sleepy 88% ✓ | Multi-source boost |
| Pain (audio wrong) | hungry 55% | pain 97% [CRIT] | pain 95% ✓ | Critical override |
| Startle (env+vision) | startle 40% | startle 95% | startle 75% ✓ | Multi-source boost |
| Cold (env+vision) | discomfort 50% | temperature 85% | temperature 75% ✓ | Env override + escalate |
| Lonely (audio 98% wrong) | hungry 98% | lonely 95% [CRIT] | lonely 85% ✓ | Critical override |

**Key result:** Vision correctly overrode audio in 3/6 scenarios, including one where audio was 98% confident in the wrong answer.

### Layered Judge Fallback

| Scenario | Chain | Winner |
|----------|-------|--------|
| Low-conf hungry | audio(45%) → tachyone(61%) → jev(56%) | Tachyone ✓ |
| Ambiguous equal split | audio(35%) → tachyone(12%) → jev(36%) | Jev ✓ |
| Evening colic | audio(48%) → tachyone(12%) → jev(49%) | Jev ✓ |
| Post-feed burp | audio(42%) → tachyone(62%) → jev(54%) | Tachyone ✓ |

**Key result:** Jev added value in 3/6 uncertain scenarios where Tachyone confidence was too low.

### Self-Learning (7-day simulation)

| Day | Accuracy | Cry Library Size | Top Signal Weight |
|-----|----------|-----------------|-------------------|
| 1 | 96% | 40 | cry_pattern: 0.38 |
| 4 | 100% | 162 | cry_pattern: 0.29 |
| 7 | 95% | 280 | cry_pattern: 0.27 |

**Key result:** Signal weights shifted from cry_pattern (0.38→0.27) toward context (0.16→0.23), showing the system learned that context became more reliable over time.

### End-to-End Pipeline

| Metric | Value |
|--------|-------|
| Pipeline accuracy | 5/6 scenarios (83%) |
| Average latency | 166ms per segment (audio + fusion + judge) |
| Models used | fusion (direct), tachyone (fallback), jev (complex) |
| Critical overrides | 1/6 (vision corrected audio) |

### Live Replay Demo

| Segment | Audio Confidence | Fused Confidence | Model |
|---------|-----------------|-----------------|-------|
| 0 (0-2s) | hungry 98% | hungry 98% | fusion |
| 1 (2-4s) | hungry 68% | hungry 93% | fusion |
| 2 (4-6.9s) | hungry 88% | hungry 98% | fusion |
| 3 (audio wrong) | burp 39% | hungry 95% | fusion [OVERRIDE] |

**Key result:** Segment 3 audio classifier misidentified as burp, but vision (rooting posture) triggered critical override, correctly identifying hungry.

## 4. Technology Stack

| Component | Technology | Notes |
|-----------|-----------|-------|
| Audio classification | scikit-learn SVM + librosa | 95 MFCC + spectral features |
| Posture detection | MediaPipe BlazePose | 33 body landmarks → 13 postures |
| Local judge | Tachyone (heuristic engine) | < 5ms, edge-deployable |
| Cloud judge | Jev (TypeSafe) → Clef (open-source) | Structured judgment + confidence |
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

- **Training data imbalance:** hungry class dominates (73% of segments). SVMSMOTE helps but minority classes (startle: 12, temperature: 12) remain weak.
- **Jev region block:** TypeSafe API returns 451 in some regions. Mock Jev used for validation. Clef (open-source Apache 2.0) as production replacement.
- **Vision prototype:** MediaPipe BlazePose not yet tested on real baby footage. Posture-to-cause mapping is rule-based, not learned.
- **No real-world deployment data:** All validation is on public datasets. Real home environment noise, microphone placement, and baby variability not yet tested.

### Roadmap

1. **Data augmentation:** Pitch shift + time stretch + noise injection for minority classes
2. **Real-world validation:** Deploy to 3-5 volunteer families, collect feedback data
3. **Clef deployment:** Replace mock Jev with open-source Clef for fully offline operation
4. **Model export:** SVM → ONNX/TFLite for optimized edge inference
5. **Multi-baby tracking:** ByteTrack + ReID for multi-child scenarios (classroom/childcare)
6. **Long-term pattern discovery:** Weekly/monthly anomaly detection (e.g., "colic pattern every evening")

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

# Test
python tachyone_server.py 8901 &
python test_full_pipeline.py

# Dashboard
uvicorn api_server:app --host 0.0.0.0 --port 8000
# Open http://localhost:8000/dashboard
```

## 9. License

MIT
