# Hoffmann et al., 2010 - Temporal parameters of spontaneous speech in Alzheimer's disease

- **Citation**: Hoffmann, I., Nemeth, D., Dye, C. D., Pákáski, M., Irinyi, T., & Kálmán, J. (2010). Temporal parameters of spontaneous speech in Alzheimer's disease. *International Journal of Speech-Language Pathology, 12*(1), 29-34.
- **DOI / PMID**: 10.3109/17549500903137256 / PMID not recorded
- **Study design**: Cross-sectional, 4-minute spontaneous speech samples (5-8 min sessions: explain why at the clinic, recount life events, describe a hobby). Kruskal-Wallis and Mann-Whitney tests.
- **Population**: 30 probable-AD patients (DSM-IV / ICD-10), Hungarian native speakers, medication-free, plus 15 age-matched controls. **Staged by MMSE, not GDS/FAST**: mild 21-26, moderate 15-21, severe 0-14. n = 10 per AD group.

| Group (MMSE mean, SD) | Speech tempo (phonemes/s) | Articulation rate (phonemes/s) | Hesitation ratio (%) | Grammatical errors (%) |
|---|---|---|---|---|
| Control (28.6, 0.42) | 9.50 ± 1.11 | 11.48 ± 1.08 | 18.39 ± 4.74 | 4.33 ± 1.29 |
| Mild (25.8, 0.42) | 7.76 ± 1.31 | 10.84 ± 1.52 | 28.52 ± 6.32 | 5.2 ± 1.22 |
| Moderate (19.8, 1.54) | 5.34 ± 1.49 | 9.38 ± 1.70 | 43.6 ± 8.6 | 6.2 ± 1.23 |
| Severe (9.4, 2.54; range 5-12) | 5.80 ± 1.95 | 9.52 ± 1.73 | 45.44 ± 6.69 | 46 ± 20.46 |

- **Parameter(s) extracted**:
  - `verbal_fluency_mean_wpm`: **not reported directly.** Study measures phonemes/second (Table II), not words per turn. Converting needs a phonemes-per-word figure the paper does not give, so use *relative* decline only: speech tempo is 0.82x control at mild, 0.56x at moderate, 0.61x at severe (0.69x and 0.75x of the mild value for moderate and severe). Hesitation ratio (share of the 4-min sample spent in pauses >30 ms): 28.5% mild, 43.6% moderate, 45.4% severe.
  - `non_verbal_cue_probability`: not measured.
  - `factual_inversion_rate`: not measured. Grammatical error ratio (syntax/morphology errors per utterance) is a different construct and must not be used as a substitute.
  - `anomia_rate`: not measured (word-finding difficulty is only inferred from hesitation).
  - `distress_escalation_prob`: not measured.
- **Notes / caveats**:
  - Moderate and severe groups did **not** differ significantly on speech tempo (p = .80), articulation rate (p = .58) or hesitation ratio (p = .48). The paper cannot support a steep fluency drop between moderate and severe.
  - The "severe" group (mean MMSE 9.4) is not equivalent to GDS/FAST 7; MMSE-to-GDS mapping is a judgment call and must be documented wherever it is applied.
  - Hungarian speakers, n = 10 per group, single clinic. Phoneme rates may not transfer to English.
  - Grammatical errors rise sharply only in the severe group (46% vs ~5-6%).
  - MMSE correlations (Spearman, all n = 45): tempo r = .723, hesitation r = -.844, grammatical errors r = -.751, articulation rate r = .552.
- **Confidence**: low-medium (small samples, MMSE staging, non-English, units differ from the YAML).
