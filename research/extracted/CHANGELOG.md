# Prior changelog

Changes to `config/statistical_priors.yaml`, per the workflow in `research/README.md`.

## 2026-11-04 - Issue #14: replace placeholder provenance

Every `source` now starts with `CITED:`, `ASSUMPTION:` or `PLACEHOLDER:`. Only values a study supports were changed.

| Field | Old value | New value | New provenance |
|---|---|---|---|
| `stage_5.anomia_rate` | 0.30 | 0.305 | CITED: Silagi_2015 (any naming error, moderate AD, CDR 2) |
| `stage_6.anomia_rate`, `stage_7.anomia_rate` | 0.50, 0.75 | unchanged | ASSUMPTION: extrapolated from the Silagi_2015 trend; study excluded MMSE <= 12 |
| `verbal_fluency_mean_wpm` (stages 5-7) | unchanged | unchanged | ASSUMPTION: Hoffmann_2010 supports direction only (phonemes/s, MMSE staging, moderate and severe not significantly different) |
| `distress_escalation_prob` (stages 5-7) | unchanged | unchanged | ASSUMPTION: anchored to Steinberg_2006 (OR 2.42 per CDR point) and Steinberg_2008 (13-24% monthly prevalence) |
| `factual_inversion_rate` (stages 5-7) | unchanged | unchanged | PLACEHOLDER: no direct source; Steinberg_2008 delusion prevalence is a loose proxy |
| `non_verbal_cue_probability` (stages 5-7) | unchanged | unchanged | PLACEHOLDER: no source found |
| `distress_transition_matrix` | unchanged | unchanged | ASSUMPTION; stationary agitated share 19.5% is inside Steinberg_2008's 13-24% |

Also in this change: YAML `version` bumped to 2; `research/extraction_log.csv` records every extracted number with a DOI link; `python -m cli.app priors-status` reports provenance and exits 1 while any prior is not `CITED`; `pytest -m research_gate` fails for the same reason and is deselected from the default run. `ASSUMPTION` counts as not sourced by the guard.

### Why most values did not change

The sources found cover mild to moderate AD, are staged by MMSE or CDR rather than GDS/FAST, and report prevalence or test scores rather than per-turn probabilities. Replacing stage 6-7 numbers from them would swap an honest placeholder for a falsely cited value. Gaps still open: late-stage speech output, stage-specific confabulation or misidentification rates, and non-verbal cue frequency.

### Sources reviewed but not used for values

- Johnson, Hashtroudi & Lindsay (1993), source monitoring: theory only, no dementia data.
- El Haj et al. (2020), false memory in AD: narrative review, mild to moderate only. Design note: confabulation in AD is mostly provoked by questions and only occasionally spontaneous.
- Anatchkova et al. (2019), agitation systematic review: ranges only (5-88% prevalence), no per-stage values.
- Kim & Park (2017), person-centered care meta-analysis: intervention effect sizes, not epidemiology.
