# Steinberg et al., 2008 - Point and 5-year period prevalence of neuropsychiatric symptoms in dementia (Cache County Study)

- **Citation**: Steinberg, M., Shao, H., Zandi, P., Lyketsos, C. G., Welsh-Bohmer, K. A., Norton, M. C., Breitner, J. C. S., Steffens, D. C., Tschanz, J. T., & Cache County Investigators. (2008). Point and 5-year period prevalence of neuropsychiatric symptoms in dementia: the Cache County Study. *International Journal of Geriatric Psychiatry, 23*(2), 170-177.
- **DOI / PMID**: 10.1002/gps.1858 / PMID not recorded
- **Study design**: Population-based longitudinal cohort, incident dementia cases. Neuropsychiatric Inventory (NPI), caregiver interview, symptoms present in the past month. Assessed at baseline, 1.5, 3.0, 4.1 and 5.3 years.
- **Population**: 408 community-dwelling people with incident dementia in Cache County, Utah (about 63% AD, 11% vascular, 7% mixed). Staged by CDR at baseline: mostly mild (74-86%), moderate 11-14%, severe (CDR 3-5) 3.5-12%. Very little cholinesterase-inhibitor use (under 9%). Attrition was high (fewer than 10% of baseline participants had a 5.3-year visit).

| NPI agitation/aggression | Baseline | 1.5 y | 3.0 y | 4.1 y | 5.3 y |
|---|---|---|---|---|---|
| Mean severity among those with symptom (frequency x severity, scale 1-12) | 3.4 (2.6) | 3.1 (2.4) | 3.8 (2.6) | 3.2 (2.1) | 2.5 (2.3) |

- Point prevalence of agitation/aggression fluctuated between 13% and 24% across the five visits (per-visit values are in Figure 2 of the paper, not transcribed here).
- Baseline agitation/aggression: 34 of 172 (19.9%) in people with no follow-up visit vs 24 of 236 (10.2%) in people with at least one follow-up (p = .01). The first group was older and had more severe dementia. Pooled baseline is about 14% (58/408, derived here).
- **Parameter(s) extracted**:
  - `distress_escalation_prob`: only an indirect anchor. Monthly prevalence of 13-24% says how many patients show agitation/aggression within a month; it is not a per-turn escalation probability.
  - `distress_transition_matrix`: consistency check only. The matrix in `config/statistical_priors.yaml` has a stationary agitated share of 19.5% (computed by power iteration; calm 36.8%, confused 25.4%, withdrawn 18.3%), inside the 13-24% range. This is a plausibility check, not a calibration: simulator "agitated" is a per-turn state, NPI agitation is any occurrence in a month.
  - `verbal_fluency_mean_wpm`, `non_verbal_cue_probability`, `factual_inversion_rate`, `anomia_rate`: not measured.
- **Notes / caveats**:
  - Not stratified by stage in the paper, so it cannot separate stages 5, 6 and 7. Use together with the severity odds ratio in `steinberg_2006.md`.
  - Pooled dementia types, not AD only. The 2006 paper reports AD diagnosis lowers agitation odds (OR 0.58), so these figures may overstate AD-only prevalence.
  - Caregiver-reported, community-dwelling, Utah population older and less diverse than the US overall.
  - Attrition likely makes later-visit prevalences underestimates (the authors say so).
- **Confidence**: low for this project's per-stage use; medium as an overall prevalence range.
