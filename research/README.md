# Literature Extraction Template

This directory is the bridge between published research and
`config/statistical_priors.yaml`. Every prior used by the symptom sampler
should be traceable back to an entry here.

## Workflow

1. Find a relevant paper (PubMed, ACM Digital Library, or similar) reporting
   quantitative data on speech/behavioral markers in moderate-to-severe
   dementia (GDS/FAST stages 5-7).
2. Copy the template below into `research/extracted/<first-author>_<year>.md`.
3. Fill in every field. If a paper does not report a needed statistic
   directly, note how you derived or approximated it (e.g., "estimated from
   Figure 2 boxplot median").
4. Update the corresponding field in `config/statistical_priors.yaml`,
   replacing the `source` placeholder with `"<first-author>_<year>"` and the
   `value`/`mean`/`std` with the extracted numbers.
5. Record the change in `research/extracted/CHANGELOG.md` (create it if it
   does not exist) with the date, field changed, old value, and new value.

## Extraction Template

```markdown
# <Author et al., Year> - <Short Title>

- **Citation**: Full APA/AMA citation.
- **DOI / PMID**:
- **Study design**: (e.g., cross-sectional observational, N=45 patients)
- **Population**: Diagnostic criteria used, stage classification scheme
  (GDS/FAST/CDR), sample size per stage.
- **Parameter(s) extracted**:
  - `verbal_fluency_mean_wpm`: value, units, stage, measurement method
  - `non_verbal_cue_probability`: value, definition of "non-verbal cue" used
    by the study
  - `factual_inversion_rate`: value, definition (e.g., confabulation rate,
    misattribution rate)
  - `anomia_rate`: value, instrument used (e.g., Boston Naming Test)
  - `distress_escalation_prob`: value, context (e.g., reaction to
    orientation questioning)
- **Notes / caveats**: Sample size limitations, population differences,
  measurement caveats that affect how directly this generalizes to the
  simulation's priors.
- **Confidence**: low / medium / high, given sample size and study quality.
```

## Design Principle

The simulation code (`src/behavior/symptom_sampler.py`) never hard-codes a
numeric prior. It always reads from `statistical_priors.yaml`. This means:

- Updating a citation is a config change, not a code change.
- The provenance of every number in a generated transcript can be traced
  back to a specific paper (or flagged as a placeholder pending one).
- Multiple competing priors (e.g., different studies disagreeing on anomia
  rate) can be captured as alternate YAML profiles and A/B tested.

## Current Status

All entries in `config/statistical_priors.yaml` are placeholders pending
extraction. See the `source` field of each prior. Contributions adding real
extracted values (with citations) are highly welcome.
