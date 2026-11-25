# OOD gate report — robust feature bank

Date: 2026-11-25

Classifier features: RMS loudness + per-recording CMVN.
OOD features: raw RMS, no CMVN.

- Score: minimum per-class Mahalanobis distance after standardization
- Calibration: GroupKFold by recording (unseen recordings)
- Default threshold: 85th percentile of in-distribution
  calibration scores
- Production detector: 4993 segments, 5 classes
- Threshold: 11.748

## LOCO detection

| Held-out corpus | n | AUC | TPR@10% FPR | TPR@15% FPR | Gate rate | Trusted n | Trusted acc |
|---|---:|---:|---:|---:|---:|---:|---:|
| donate | 2492 | 1.000 | 100% | 100% | 100% | 0 | n/a |
| dbl | 482 | 0.924 | 79% | 85% | 85% | 72 | 17% |
| dynamic | 2019 | 0.915 | 67% | 88% | 88% | 234 | 39% |
| **mean** | | **0.946** | **82%** | **91%** | **91%** | **306** | |

## Interpretation

- This gate is paired with the channel-robust classifier but scores
  OOD on raw RMS features. CMVN removes the recording-level offset
  that the gate relies on; scoring CMVN features drops mean LOCO
  AUC from ~0.95 to ~0.70.
- Do not mix this detector with the peak-normalized production model.
- CMVN is transductive within a recording. Live inference needs two
  segments before the rolling buffer is active.
- A high gate rate on a new corpus is expected: CMVN recovers part of
  the channel shift, not the label/recording-protocol shift.
