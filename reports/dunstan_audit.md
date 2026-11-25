# Dunstan label-semantics audit

Source definitions are from the local `data/infantcry-dbl-v2/README.md` and `metadata.csv`.

## 1. What the labels actually mean

| DBL | Corpus definition | Our mapping | Fit |
|---|---|---|---|
| eairh | lower-wind / abdominal discomfort (straining, lower gas) | pain | **over-strong** |
| eh | upper-wind / need to burp (burp reflex) | burp | direct |
| heh | physical discomfort (skin-related stimulus) | discomfort | direct |
| neh | hunger (sucking reflex against the soft palate) | hungry | direct |
| owh | sleepiness (yawn-like rounding) | sleepy | direct |

`eairh` is defined as lower-wind/abdominal discomfort. Mapping it to `pain` conflates gas discomfort with sharp pain; `discomfort` is the closer label. `heh` is skin-related discomfort and `eh` is upper-wind/burp, both direct matches.

## 2. Corpus composition

- Clips: 1551 (54.4 min)
- Dunstan-Core: 337 clips, {'eairh': 60, 'eh': 72, 'heh': 57, 'neh': 72, 'owh': 76}
- InfantCry-1214: 1214 clips, {'eairh': 110, 'eh': 208, 'heh': 111, 'neh': 145, 'owh': 640}
- Sources: {'dunstan_video': 337, 'cryceleb_likely': 231, 'online_or_volunteer': 237, 'unknown': 746}
- source_confirmed: {'TRUE': 337, 'FALSE': 1214}

Label inconsistencies recorded in metadata:

- `1214 COLLCTED DATA/eh/000543_M_heh.wav`: filename_label=heh; folder_label=eh; folder_is_authoritative

`metadata.csv` paths do not resolve for 1214 rows (example: `1214 COLLCTED DATA/eairh/1028_M_eairh.wav`). The archive folder is `1214 Collected Data` while the CSV says `1214 COLLCTED DATA`; the loader rewrites the prefix.

Tier 1 is the 337 studio clips used by the previous training/eval; Tier 2's 1,214 heterogeneous clips were unused. All Tier-2 `source_confirmed` values are FALSE and 746 of them have source `unknown`.

## 3. Are the five sounds separable?

| Experiment | n | Acc | Balanced | Majority |
|---|---:|---:|---:|---:|
| cv_Dunstan-Core | 482 | 78.6% | 78.2% | 26.6% |
| cv_InfantCry-1214 | 1559 | 86.9% | 80.0% | 47.0% |
| Dunstan-Core->InfantCry-1214 | 1559 | 56.5% | 67.5% | 47.0% |
| InfantCry-1214->Dunstan-Core | 482 | 76.8% | 76.8% | 26.6% |

Within-corpus separation is the ceiling for any mapping: if the sounds are not separable even inside one tier, the label mapping is not the bottleneck. Cross-tier rows show how much recording setup alone changes the result.

## 4. Mapping A vs B, trained on Donate+Dynamic

| Mapping / test tier | n | Acc | Balanced |
|---|---:|---:|---:|
| A_eairh_pain/Dunstan-Core | 482 | 20.3% | 19.3% |
| A_eairh_pain/InfantCry-1214 | 1559 | 15.2% | 20.4% |
| A_eairh_pain_plus_tier2/Dunstan-Core | 482 | 70.5% | 70.3% |
| B_eairh_discomfort/Dunstan-Core | 482 | 21.0% | 24.4% |
| B_eairh_discomfort/InfantCry-1214 | 1559 | 15.1% | 25.2% |
| B_eairh_discomfort_plus_tier2/Dunstan-Core | 482 | 71.6% | 71.9% |

`eairh` predictions: {np.str_('hungry'): 232, np.str_('sleepy'): 3, np.str_('discomfort'): 4, np.str_('pain'): 2}

## 5. Acoustic profile by Dunstan class

| Class | Segments | Centroid | RMS | F0 mean |
|---|---:|---:|---:|---:|
| eairh | 241 | 2770.8 | 0.1418 | 389.5 |
| eh | 414 | 1783.5 | 0.1009 | 336.4 |
| heh | 198 | 2062.6 | 0.1196 | 400.5 |
| neh | 328 | 2327.0 | 0.1684 | 416.2 |
| owh | 860 | 2640.0 | 0.1262 | 357.7 |

## Conclusion

1. The mapping is mostly direct, but `eairh -> pain` is stronger than the corpus's own definition (lower-wind / abdominal discomfort). It is retained in production because DBL's `eairh` segments are 36% of all pain training data and re-mapping to `discomfort` costs pain recall; see `reports/eairh_mapping.md` for the A/B benchmark.
2. The corpus is five *reflex sounds*, not five verified causes. Even a perfect DBL classifier is not a cause diagnosis.
3. Only Tier 1 was used before. Tier 2 adds 1,214 clips but is unconfirmed and heterogeneous; it is useful for domain-shift stress tests, not as clean ground truth.
4. Any cross-corpus number should be reported with the mapping and tier, because both change the label and the domain.
