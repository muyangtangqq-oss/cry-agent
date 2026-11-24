# Leave-one-corpus-out report

Date: 2026-11-24

## Setup

- Classes: burp, discomfort, hungry, pain, sleepy
- Segments: 4993
- Per corpus: {'donate': 2492, 'dbl': 482, 'dynamic': 2019}
- Model: StandardScaler + SVMSMOTE + SVM-RBF
- Group CV groups by source recording; LOCO holds out an entire
  corpus and evaluates only the five shared labels.

## Results

| Evaluation | n | Accuracy | Balanced | Majority baseline |
|---|---:|---:|---:|---:|
| Group 5-fold (recording) | 4993 | 76.6% | 41.9% | 74.5% |
| LOCO donate | 2492 | 79.6% | 22.6% | 83.3% |
| LOCO dbl | 482 | 20.3% | 19.3% | 21.0% |
| LOCO dynamic | 2019 | 21.3% | 36.2% | 76.4% |
| **LOCO mean** | | **40.4%** | **26.0%** | 74.5% |

## Per-class recall

| Evaluation | burp | discomfort | hungry | pain | sleepy |
|---|---:|---:|---:|---:|---:|
| Group 5-fold (recording) | 31% | 19% | 93% | 40% | 26% |
| LOCO donate | 13% | 0% | 95% | 0% | 5% |
| LOCO dbl | 0% | 0% | 93% | 1% | 2% |
| LOCO dynamic | 74% | 38% | 18% | 33% | 18% |

## Confusion matrices

### LOCO donate

| true / pred | burp | discomfort | hungry | pain | sleepy |
|---|---:|---:|---:|---:|---:|
| burp | 6 | 0 | 40 | 0 | 0 |
| discomfort | 5 | 0 | 142 | 0 | 0 |
| hungry | 102 | 0 | 1970 | 0 | 4 |
| pain | 1 | 0 | 89 | 0 | 0 |
| sleepy | 2 | 0 | 124 | 0 | 7 |

### LOCO dbl

| true / pred | burp | discomfort | hungry | pain | sleepy |
|---|---:|---:|---:|---:|---:|
| burp | 0 | 0 | 77 | 0 | 3 |
| discomfort | 0 | 0 | 66 | 0 | 0 |
| hungry | 0 | 2 | 94 | 3 | 2 |
| pain | 0 | 4 | 101 | 1 | 1 |
| sleepy | 0 | 4 | 121 | 0 | 3 |

### LOCO dynamic

| true / pred | burp | discomfort | hungry | pain | sleepy |
|---|---:|---:|---:|---:|---:|
| burp | 40 | 6 | 1 | 1 | 6 |
| discomfort | 46 | 50 | 19 | 9 | 8 |
| hungry | 637 | 342 | 272 | 138 | 153 |
| pain | 45 | 16 | 6 | 34 | 1 |
| sleepy | 83 | 40 | 16 | 16 | 34 |

## Interpretation

- Group CV: **76.6%** accuracy / **41.9%** balanced.
- LOCO mean: **40.4%** accuracy / **26.0%** balanced.
- The gap is the cost of changing collection setup. A large gap
  means the model is learning corpus-specific cues, not a
  transferable cry signature.
- Majority baseline reaches **74.5%** LOCO accuracy. The model is at **40.4%**, so it is not merely weak; on all three held-out corpora it is worse than always
  predicting the training majority class.
- Even group CV (76.6%) is only +2.1 pp over its majority baseline; most of the apparent skill is
  class-frequency and corpus identity.
- Do not report in-corpus accuracy as cross-dataset generalization.
- Startle/temperature/lonely are excluded: they exist only in one
  corpus and cannot be evaluated cross-corpus.

## What to do next

1. Re-check label semantics across corpora (Dunstan `eh`/`eairh` vs
   Donate `burping`/`belly_pain` may not be the same construct).
2. Add channel/room augmentation and per-recording normalization,
   then re-run LOCO; do not accept a change that only lifts CV.
3. Try domain-invariant features or CORAL/adversarial training.
4. Collect a small in-family calibration set and adapt per household.
5. Add an out-of-distribution gate so a new corpus escalates instead
   of returning a confident but wrong cause.
