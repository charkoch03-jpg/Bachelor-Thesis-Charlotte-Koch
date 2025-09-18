# Exploratory Analyses — Timecourse Analysis

This folder contains scripts for timecourse analyses of pupil size across Free Viewing and Cue Task experiments. Both extract pupil size at multiple timepoints, add trial metadata, and run linear mixed models (LMMs) over time.

---

## Files

### `lmmTimecourse_freeViewing.py`
- Extracts pupil size at ~100 ms intervals for each trial.
- Metadata: participant, current valence, previous valence.
- Fits LMMs: `pupilSize_i ~ valence * previous_valence + (1|participant)` per timepoint.
- Plots p-values of `valence` and `previous_valence` over time relative to picture onset.

### `lmmTimecourse_cueTask.py`
- Extracts pupil size at 0–1500 ms (50 ms steps) per trial.
- Metadata: participant, valence, previous_valence, task condition.
- Fits LMMs: `pupil_tp ~ valence + previous_valence + condition + (1|participant)` per timepoint.
- Plots p-values of fixed effects over time.

---
