# Steinberg et al., 2006 - Risk factors for neuropsychiatric symptoms in dementia (Cache County Study)

- **Citation**: Steinberg, M., Corcoran, C., Tschanz, J. T., Huber, C., Welsh-Bohmer, K., Norton, M. C., Zandi, P., Breitner, J. C. S., Steffens, D. C., & Lyketsos, C. G. (2006). Risk factors for neuropsychiatric symptoms in dementia: the Cache County Study. *International Journal of Geriatric Psychiatry, 21*(9), 824-830.
- **DOI / PMID**: 10.1002/gps.1567 / PMID not recorded
- **Study design**: Population-based cohort. NPI (caregiver interview, symptom present in the past month) at baseline for 328 people with prevalent dementia, re-administered about 18 months later to 184. Generalized estimating equations model the probability that each symptom is present as a function of eight risk factors, each adjusted for the others.
- **Population**: Cache County, Utah. Baseline CDR among those with and without follow-up: mild (CDR 1) 30-45%, moderate (CDR 2) 27-29%, severe (CDR 3-5) 27-43%. Roughly 59-66% AD.
- **Parameter(s) extracted** (agitation/aggression odds ratios, Table 2, 95% CI in brackets):
  - **Dementia severity, per one-point CDR increase: OR 2.42 (1.81-3.23), p < .01.** Also hallucinations OR 1.55 (1.14-2.12); delusions not significant (1.22, 0.93-1.59); depression decreased (0.75, 0.58-0.99).
  - **AD diagnosis vs non-AD dementia: OR 0.58 (0.35-0.95).**
  - **General medical comorbidity (GMHR point increase): OR 1.76 (1.01-2.81).**
  - Gender, age, education, APOE e4 and time of observation: not significant for agitation/aggression.
  - `distress_escalation_prob`: no direct value. The per-stage multiplier can scale the overall prevalence from `steinberg_2008.md` across stages, but any such derived stage-wise figure is an assumption-labelled estimate and must be marked that way in the YAML `source` field.
  - `verbal_fluency_mean_wpm`, `non_verbal_cue_probability`, `factual_inversion_rate`, `anomia_rate`: not measured.
- **Notes / caveats**:
  - Outcome is presence/absence of the symptom in the past month; frequency and severity were not modeled.
  - Staging is CDR (0-5), not GDS/FAST. Mapping CDR 3-5 to GDS/FAST 6-7 is a judgment call to document where used.
  - The OR is per CDR point and assumes a log-linear trend; extrapolating beyond the observed range is a modeling choice.
  - Caregiver-reported; community sample in Utah; those with follow-up were younger and milder than those without.
- **Confidence**: medium for the direction and size of the severity effect; low for converting it to per-turn probabilities.
