# Exploratory Analyses — Cheater vs Non-Cheater

This folder contains scripts to classify trials as cheater/non-cheater, plot pupil size timecourses, and run linear mixed models (LMMs) separately for cheaters and non-cheaters in the cue task.

---

## Files

### `labelCheater_cueTask.py`
- Determines cheater trials based on gaze position at picture onset relative to a the trial specific center.
- Saves `cheater.mat` (0 = non-cheater, 1 = cheater).

### `pupilPlots_cheater.py` / `pupilPlots_noncheater.py`
- Extracts pupil size from dominant eye per trial.
- Computes mean ± CI across participants.
- Plots average pupil size over time for:
  - Current valence (pos/neg)
  - Previous valence (pos/neg)
  - Task condition (congruent/incongruent)
- Separate scripts for cheater vs non-cheater trials.

### `lmm_cheater.py` / `lmm_noncheater.py`
- Builds trial-level DataFrames with pupil size at -1000ms to +1400 ms (100 ms steps) and participantID, valence, previous_valence, task condition.
- Fits LMMs per timepoint: `pupil_tp ~ valence + previous_valence + condition + (1|participant)`.
- Collects and plots p-values of fixed effects over time.
- Separate models for cheater and non-cheater trials.

---

