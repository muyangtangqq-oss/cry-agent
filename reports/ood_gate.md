# OOD gate report

Date: 2026-11-24

## Method

- Score: minimum per-class Mahalanobis distance after standardization
- Calibration: GroupKFold by recording (unseen recordings)
- Default threshold: 85th percentile of in-distribution
  calibration scores
- Production detector: 5053 segments, 8 classes
- Threshold: 11.784

## LOCO detection

| Held-out corpus | n | AUC | TPR@10% FPR | TPR@15% FPR | Gate rate | Trusted n | Trusted acc |
|---|---:|---:|---:|---:|---:|---:|---:|
| donate | 2492 | 1.000 | 100% | 100% | 100% | 0 | n/a |
| dbl | 482 | 0.920 | 78% | 84% | 84% | 77 | 19% |
| dynamic | 2019 | 0.912 | 66% | 86% | 86% | 280 | 16% |
| **mean** | | **0.944** | **81%** | **90%** | **90%** | **357** | |

## Interpretation

- The gate is a safety mechanism, not a classifier. It flags inputs
  that are far from every training class and turns them into
  `unknown` / low-trust decisions.
- At 15% false-alarm rate it catches about 90% of held-out corpus
  segments on average; DynamicSuperb is the hardest corpus.
- The few held-out segments that pass the gate are still only
  16-19% accurate on DBL/DynamicSuperb. Per-sample gating alone is
  not enough; the pipeline also has a session-level gate (8 of the
  last 12 clips must exceed the threshold to mark the session OOD).
- A gate cannot recover the lost information. It buys honesty, not
  accuracy. Domain adaptation and family data are still required.
- Do not interpret 'not flagged' as 'trustworthy' on a new corpus.
