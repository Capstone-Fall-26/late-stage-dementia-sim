# Mitchell, 2015 - Advanced Dementia (NEJM Clinical Practice)

- **Citation**: Mitchell, S. L. (2015). Advanced dementia. *New England Journal of Medicine, 372*(26), 2533-2540.
- **DOI / PMID**: 10.1056/NEJMcp1412652 / PMID not recorded
- **Study design**: Clinical practice review (not a primary study). Summarizes the CASCADE cohort (323 nursing home residents with advanced dementia, 18 months) and the hospice FAST criteria.
- **Population**: Advanced dementia, GDS 7 / FAST 7.
- **Parameter(s) extracted**:
  - `verbal_fluency_mean_wpm` (stage 7 ceiling): Table 1 gives the FAST stage definitions used for hospice eligibility (attributed to the National Hospice Organization). **FAST 7a: speech limited to fewer than six intelligible words during an average day. 7b: one intelligible word during an average day.** This is a staging definition, not a measurement, but it is on the project's own GDS/FAST scale. It bounds any single turn at stage 7 to 5 words or fewer. The mean and std in the YAML remain modeling choices.
  - FAST 6a-6e (stages before 7) concern dressing, bathing, toileting and continence, not speech, so they say nothing about stage 5 or 6 fluency.
  - Other parameters: not measured. CASCADE complications (eating problems 86%, febrile episodes 53%, pneumonia 41%) are outside the simulator's priors.
- **Notes / caveats**:
  - The limit is per *day*; the simulator samples per *turn*. A session of several turns would already exceed five words at the old stage 7 mean of 6 words per turn, which is why the ceiling is applied per turn.
  - The simulator's stage 7 patient is therefore nearly mute by definition. Other FAST 7 sub-stages (7c cannot walk, 7d cannot sit up, 7e cannot smile, 7f cannot hold up the head) mean parts of the stage 7 non-verbal cue bank (for example a smile, turning away) fit only the earlier sub-stages.
  - A clinical review citing the FAST table, not the original FAST publication (Reisberg). Worth checking against it before publishing.
- **Confidence**: high for the staging definition itself; the per-turn conversion is an assumption.
