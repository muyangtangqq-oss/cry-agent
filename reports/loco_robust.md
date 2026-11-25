# LOCO: channel/gain robustness

Channel augmentation = random smooth EQ, soft band limit, short
reverb, and gain + soft saturation. Augmented copies are training-
only. CMVN subtracts each recording's mean feature vector; files
with a single segment are left unchanged.

| Mode | Group acc / bal | Donate | DBL | Dynamic | LOCO mean | vs base |
|---|---:|---:|---:|---:|---:|---:|
| base | 76.6% / 41.9% | 79.6% / 22.6% | 20.3% / 19.3% | 21.3% / 36.2% | **40.4% / 26.0%** | +0.0 / +0.0 |
| rms | 76.3% / 42.2% | 79.2% / 22.6% | 20.7% / 19.8% | 19.3% / 35.9% | **39.8% / 26.1%** | -0.6 / +0.0 |
| cmvn | 68.2% / 35.4% | 81.0% / 27.8% | 19.9% / 19.4% | 52.4% / 29.0% | **51.1% / 25.4%** | +10.7 / -0.7 |
| rms_channel | 77.1% / 42.6% | 77.5% / 27.8% | 20.5% / 19.4% | 25.5% / 40.0% | **41.2% / 29.1%** | +0.8 / +3.0 |
| cmvn_channel | 72.2% / 40.5% | 84.3% / 39.0% | 20.3% / 19.3% | 53.5% / 50.5% | **52.7% / 36.3%** | +12.3 / +10.2 |

Weighted majority baseline across LOCO corpora: 74.5%.

CMVN coverage (segments in recordings with >= 2 segments): donate 2492/2492 (100.0%), dbl 250/482 (51.9%), dynamic 2010/2019 (99.6%).

Caveats: CMVN and channel banks are generated once over the whole
dataset, but copies are only used for training indices in each fold.
CMVN uses each recording's own unlabeled segments (transductive);
a live deployment needs a rolling session buffer.

## Full-data model check

Training the complete robust model (RMS loudness + CMVN + one channel copy
per segment + SMOTE) reaches 69.0% accuracy / 30.3% balanced in segment-level
5-fold CV. The current production model reports 79.4% / 52.3% under its own
augmentation CV, but that run used SVMSMOTE and no CMVN, so the two CV
numbers are not directly comparable. The fair comparison is the LOCO table
above, where every mode uses the same folds and pipeline.

The robust model is written to `~/.config/cry-agent/models_robust/` and is
**not promoted**. It has no OOD detector yet, so deploying it as-is would
disable the containment gate. Use it only after fitting an OOD detector on
the same RMS + CMVN feature bank.
