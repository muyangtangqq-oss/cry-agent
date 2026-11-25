# `eairh` mapping decision

Dunstan defines `eairh` as lower-wind / abdominal discomfort, but the
label also feeds training: the 107 DBL `eairh` segments are 36% of all
`pain` segments. This compares keeping the current mapping (A) with
re-mapping to `discomfort` (B) on the shared five-class benchmark.

| Evaluation | A acc / bal | B acc / bal |
|---|---:|---:|
| group5 | 76.6% / 41.9% | 76.7% / 42.2% |
| LOCO donate | 79.6% / 22.6% | 80.0% / 22.7% |
| LOCO dbl | 20.3% / 19.3% | 21.0% / 24.4% |
| LOCO dynamic | 21.3% / 36.2% | 16.9% / 30.0% |
| **LOCO mean** | **40.4% / 26.0%** | **39.3% / 25.7%** |

Pain recall:

| Evaluation | A | B |
|---|---:|---:|
| group5 | 40% | 28% |
| LOCO donate | 0% | 0% |
| LOCO dbl | 1% | 0% |
| LOCO dynamic | 33% | 0% |

Discomfort recall:

| Evaluation | A | B |
|---|---:|---:|
| group5 | 19% | 32% |
| LOCO donate | 0% | 0% |
| LOCO dbl | 0% | 2% |
| LOCO dynamic | 38% | 61% |

## Decision

Keep A (`eairh -> pain`). B gains 5.1 pp balanced accuracy on DBL
alone, but loses 1.1 pp LOCO accuracy, drops DynamicSuperb pain
recall from 33% to 0%, and cuts overall group-CV pain recall from
41% to 28%. `pain` is the safety-critical class, so a semantic
cleanup that erases its third-largest training source is a bad
trade at 107 segments.

The mapping is defensible under our own `pain` rubric, which already
includes colic: infant colic is lower-wind/gas pain, not only acute
sharp pain. The rubric is now explicit about that. Revisit only if a
real `gas` class with enough data is added end-to-end.

Caveats: segment-level cache, one seed, five shared classes only,
and DBL is one of the training corpora in these splits.
