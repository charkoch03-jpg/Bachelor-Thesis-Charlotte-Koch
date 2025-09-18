# Exploratory Analyses — First Fixation Side

This folder contains scripts to separate trials by **first fixation side** (left vs right), plot pupil size timecourses, and run linear mixed models (LMMs) at 500 ms post-picture onset.

---

## Files

### `pupilPlot_leftEye.py` / `pupilPlot_rightEye.py` / `pupilPlot_averageEye.py`
- Extracts pupil size per trial:
  - `leftEye` / `rightEye` / average of both eyes.
- Computes mean ± SEM across participants.
- Plots average pupil size over time by **first fixation side** (`0 = left`, `1 = right`).
- Adds vertical reference lines at:
  - **0 ms** (picture onset)
  - **500 ms** post-picture onset.

### `lmm_leftEye.py` / `lmm_rightEye.py` / `lmm_averageEye.py`
- Builds trial-level DataFrames with **pupil size at 500 ms** and metadata:
  - `participant`, `valence`, `previous_valence`, `condition`, `side`
- Converts relevant columns to categorical.
- Prints descriptive stats by `side`.
- Fits LMM:  
  `pupil500 ~ valence + previous_valence + condition + side + (1|participant)`  
  using **pymer4**.
