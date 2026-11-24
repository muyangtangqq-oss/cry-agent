# Minority-class augmentation — CV report

Date: 2026-11-24

## Setup

- Data: Donate-a-Cry + InfantCry-DBL + DynamicSuperb
- Segments: 5,053 (2 s, 16 kHz)
- Features: 95 (MFCC / delta / spectral / F0)
- Split: 5-fold stratified CV, same folds for baseline and augmented
- Baseline: SVM-RBF + SVMSMOTE
- Augmented: same classifier, plus time-domain variants generated **only from
  training-fold sources** (pitch ±3 st, stretch 0.9–1.1, gain −6…+3 dB,
  noise SNR 15–35 dB)

Augmentation policy:

- target 300 segments/class
- at most 40 variants per source recording
- generated: pain 299, burp 180, lonely 288, startle 288, temperature 288

## Results

| Metric | Baseline | Augmented | Delta |
|--------|----------|-----------|-------|
| Accuracy | 79.5% | 79.4% | −0.1 pp |
| Balanced accuracy | 46.0% | **52.3%** | **+6.3 pp** |

| Class | Support | Baseline recall | Augmented recall |
|-------|---------|-----------------|------------------|
| burp | 180 | 40.0% | 43.3% |
| discomfort | 345 | 33.6% | 30.1% |
| hungry | 3,719 | 93.4% | 93.0% |
| lonely | 36 | 27.8% | 33.3% |
| pain | 299 | 46.5% | 52.8% |
| sleepy | 450 | 43.6% | 41.1% |
| startle | 12 | 33.3% | 58.3% |
| temperature | 12 | 50.0% | 66.7% |

DBL regression check (same corpus, held-in):

- previous model: 322/337 = 95.5%
- augmented model: 327/337 = 97.0%

## Interpretation

- The intended effect is visible: balanced accuracy +6.3 pp, with the weakest
  classes (startle +25 pp, temperature +17 pp, pain +6 pp) improving.
- Overall accuracy is flat because hungry dominates the corpus; this tradeoff
  is expected.
- Discomfort and sleepy recall moved slightly down. The augmented model is not
  universally better; it is better calibrated toward minority classes.
- The augmented model was promoted to `~/.config/cry-agent/models`; the
  previous files are kept as `*.pre_aug.bak`.

## Caveats

- CV is segment-level. Different 2 s segments from the same recording can land
  in different folds, so these numbers are an upper bound, not real-home
  generalization.
- Startle and temperature still come from only ~12 segments. Augmentation
  cannot invent acoustic diversity that is not in the source recordings.
- A clean leave-one-corpus-out and a real-home pilot are still required.
- Follow-up LOCO result: **40.4% accuracy / 26.0% balanced**, below the
  74.5% majority baseline. Augmentation improved within-corpus minority
  recall but did not fix cross-corpus domain shift. See `loco_cv.md`.
