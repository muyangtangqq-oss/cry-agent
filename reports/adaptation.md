# Few-shot family calibration (leave-one-corpus-out)

Each corpus is held out as a simulated new family. Calibration uses 30% of its recordings, split by recording; test uses the rest. Values are mean over 5 seeds.

| Method | Accuracy | Balanced |
|---|---:|---:|
| auto | 57.6% | 44.1% |
| coral | 54.4% | 37.1% |
| coral-proto-10 | 28.3% | 30.9% |
| coral-proto-20 | 27.3% | 29.8% |
| proto-10 | 29.2% | 29.7% |
| proto-20 | 28.6% | 28.6% |
| coral-proto-5 | 25.5% | 28.1% |
| prior | 31.4% | 27.5% |
| blend-10 | 41.1% | 26.9% |
| blend-20 | 41.1% | 26.8% |
| proto-5 | 25.3% | 26.7% |
| blend-5 | 41.0% | 26.5% |
| global | 40.5% | 26.3% |

## Per corpus

### donate

| Method | Accuracy | Balanced |
|---|---:|---:|
| auto | 71.3% | 47.2% |
| coral | 71.3% | 47.2% |
| prior | 62.2% | 26.6% |
| proto-10 | 21.7% | 24.3% |
| coral-proto-10 | 22.6% | 24.2% |
| global | 79.9% | 23.0% |
| blend-5 | 79.7% | 23.0% |
| coral-proto-20 | 19.1% | 22.9% |
| blend-10 | 79.7% | 22.9% |
| blend-20 | 80.1% | 22.6% |
| proto-20 | 21.1% | 22.0% |
| proto-5 | 19.1% | 21.6% |
| coral-proto-5 | 19.8% | 21.4% |

### dbl

| Method | Accuracy | Balanced |
|---|---:|---:|
| coral-proto-20 | 41.4% | 44.5% |
| coral-proto-10 | 41.4% | 43.7% |
| auto | 40.8% | 43.4% |
| proto-20 | 41.2% | 41.5% |
| coral-proto-5 | 39.0% | 41.3% |
| proto-10 | 39.5% | 39.5% |
| proto-5 | 35.8% | 36.5% |
| blend-5 | 21.3% | 20.6% |
| prior | 21.7% | 20.5% |
| coral | 21.2% | 20.4% |
| blend-20 | 21.3% | 20.4% |
| blend-10 | 21.2% | 20.3% |
| global | 20.5% | 19.6% |

### dynamic

| Method | Accuracy | Balanced |
|---|---:|---:|
| coral | 70.5% | 43.6% |
| auto | 60.7% | 41.7% |
| blend-20 | 22.0% | 37.5% |
| blend-10 | 22.5% | 37.4% |
| global | 21.2% | 36.3% |
| blend-5 | 21.8% | 36.0% |
| prior | 10.2% | 35.4% |
| proto-10 | 26.3% | 25.1% |
| coral-proto-10 | 21.1% | 24.7% |
| proto-5 | 21.0% | 22.2% |
| proto-20 | 23.6% | 22.2% |
| coral-proto-20 | 21.5% | 21.9% |
| coral-proto-5 | 17.7% | 21.6% |

## Auto picks

| Corpus | Picks (5 seeds) |
|---|---|
| donate | coral×5 |
| dbl | coral-proto-10×1, coral-proto-20×3, proto-20×1 |
| dynamic | coral×4, global×1 |
