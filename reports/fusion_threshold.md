# Fusion direct-output threshold calibration

Labelled set: six configured baby scenarios, six real-audio
end-to-end scenarios (n=12). Elderly and pet are checked
separately at their own configured thresholds.
Fusion confidence is `vote share x evidence strength + agreement
boost`; direct output requires `confidence >= threshold` and
`escalate == false`.

| Case | Expected | Fused | Confidence | Escalate |
|---|---|---|---:|---|
| baby-1 | hungry | hungry | 98% | False |
| baby-2 | sleepy | sleepy | 98% | False |
| baby-3 | pain | pain | 95% | False |
| baby-4 | startle | startle | 98% | False |
| baby-5 | temperature | temperature | 75% | False |
| baby-6 | lonely | lonely | 85% | False |
| hungry (audio) | hungry | hungry | 98% | False |
| sleepy (audio) | sleepy | sleepy | 98% | False |
| pain (audio) | pain | pain | 95% | False |
| discomfort (audio) | temperature | temperature | 75% | False |
| burp (audio) | burp | burp | 65% | False |
| lonely (audio) | lonely | lonely | 85% | False |

## Threshold sweep

| Threshold | Direct | Direct correct | Wrong direct | Direct precision |
|---:|---:|---:|---:|---:|
| 0.70 | 11/12 | 11 | 0 | 100% |
| 0.75 | 11/12 | 11 | 0 | 100% |
| 0.80 | 9/12 | 9 | 0 | 100% |
| 0.85 | 9/12 | 9 | 0 | 100% |
| 0.90 | 7/12 | 7 | 0 | 100% |
| 0.95 | 7/12 | 7 | 0 | 100% |

## Agreement operating points

| Per-source confidence | Two sources | Three sources |
|---:|---:|---:|
| 40% | 62% | 85% |
| 50% | 73% | 98% |
| 55% | 79% | 98% |
| 60% | 85% | 98% |
| 65% | 91% | 98% |
| 70% | 95% | 98% |
| 80% | 95% | 98% |

## Cross-scenario sanity check

| Case | Threshold | Expected | Fused | Confidence | Direct |
|---|---:|---|---|---:|---|
| elderly fall | 0.85 | fall | fall | 85% | True |
| pet fear | 0.75 | fear | fear | 85% | True |

## Decision

Set the baby direct-output threshold to **0.85**. That is the
operating point for two sources at 60% or three sources at 40%;
0.80 would admit weaker two-source agreement such as 60% + 50%.
It is also the highest threshold that still keeps every baby
critical override (capped at 85%) on the direct path; 0.90 would
send them to a judge. The 75% env-override temperature cases take
a judge pass. The labelled set has no wrong direct answer at 0.75
or above, so 0.85 is conservative rather than accuracy-optimal.

Caveat: no real multimodal household data exists yet. This is a
policy calibration over constructed/real-audio scenarios, not a
home-accuracy estimate; re-calibrate after the pilot.
