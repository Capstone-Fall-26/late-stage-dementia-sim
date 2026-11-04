# Silagi et al., 2015 - Naming ability in mild to moderate Alzheimer's disease

- **Citation**: Silagi, M. L., Bertolucci, P. H. F., & Ortiz, K. Z. (2015). Naming ability in patients with mild to moderate Alzheimer's disease: what changes occur with the evolution of the disease? *Clinics, 70*(6), 423-428.
- **DOI / PMID**: 10.6061/clinics/2015(06)07 / PMC4462568
- **License**: Open access, CC BY-NC 3.0.
- **Study design**: Cross-sectional, observational, outpatient behavioral-neurology clinic (Sao Paulo, Brazil). 20-item revised Boston Naming Test (15 CERAD pictures + 5 low-frequency items); up to 20 s per item, then semantic and phonemic cues. Errors classified by type.
- **Population**: 30 mild AD, 30 moderate AD, 30 age/education-matched controls (NINCDS-ADRDA criteria). Staged by CDR (mild 0.5-1, moderate 2); enrollment required MMSE > 12 and cholinesterase-inhibitor treatment. Brazilian Portuguese speakers, about 7 years of education on average. MMSE means: control 28.6, mild 25.4, moderate 19.8.

| Mean (SD) per person, out of 20 items | Control | Mild AD | Moderate AD |
|---|---|---|---|
| Correct answers | 17.2 (2.2) | 16.4 (2.3) | 13.9 (3.9) |
| No-response errors (pure anomia) | 0.3 (0.7) | 0.9 (1.3) | 1.9 (2.5) |
| Verbal semantic paraphasia | 1.7 (1.5) | 1.6 (1.3) | 2.1 (1.3) |
| Unrelated verbal paraphasia | 0.2 (0.5) | 0.4 (0.6) | 1.0 (1.1) |
| Visual errors | 0.3 (0.5) | 0.4 (0.6) | 0.7 (0.9) |
| Circumlocution | 0.2 (0.6) | 0.3 (0.5) | 0.3 (0.8) |

Remaining error types (phonemic paraphasia, intrusion) were at or near zero in all groups.

- **Parameter(s) extracted**:
  - `anomia_rate`: two candidate definitions, both derived by dividing the per-person means above by 20 items (derived here, not stated in the paper).
    - Strict, pure anomia (no response): control 1.5%, mild 4.5%, moderate 9.5%.
    - Broad, any naming error (20 minus correct answers): control 14%, mild 18%, moderate 30.5%. The mild figure is rounded from the paper's own error counts, which sum to about 3.7 rather than 3.6.
    - Which definition `anomia_rate` should use is a project decision and should be recorded in the changelog when the YAML is updated.
  - `verbal_fluency_mean_wpm`, `non_verbal_cue_probability`, `factual_inversion_rate`, `distress_escalation_prob`: not measured.
- **Significance**: Correct answers differ across groups (ANOVA F = 10.68, p < .001); moderate AD is lower than both other groups. Pure anomia differs significantly only between moderate AD and controls. Unrelated paraphasias are higher in moderate AD than in both other groups.
- **Notes / caveats**:
  - Covers up to moderate AD (CDR 2, MMSE about 20), roughly the project's stage 5. It gives **no** evidence for stages 6-7 because patients with MMSE of 12 or below were excluded by design.
  - Single-word picture naming with easy items is not conversational word-finding failure, so rates will not transfer one-to-one to dialogue.
  - All patients were on cholinesterase inhibitors, which may blunt decline.
  - Small samples, Brazilian Portuguese, low average education, only 20 items.
  - The qualitative pattern is also useful for cue banks and prompts: semantic paraphasia is the dominant error type in every group, and error types diversify (unrelated words, visual errors) in moderate AD.
- **Confidence**: medium for stage 5 only; none for stages 6-7.
