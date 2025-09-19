# Cue Task — Pupil Main Analysis

This folder contains scripts for analyzing pupil responses in the Cue Task, including plotting trial-locked time courses and running linear mixed models (LMM).

## Files in this folder

**PupilPlots_cueTask.py**  
Prepares and visualizes pupil size data across trials and participants.  
- Loads left/right eye pupil traces based on dominant-eye info, cue side, block order, and picture sequences.  
- Computes per-trial labels for current valence, previous valence, and task condition (congruent/incongruent).  
- Splits data into groups by valence, previous valence, task, and their interactions.  
- Computes participant-level means and 95% confidence intervals.  
- Produces time-course plots for:
  - Current valence
  - Previous valence
  - Interaction: Previous × Current valence
  - Task condition
  - Interaction: Task × Current valence  
- Prints descriptive statistics of pupil size values at 500 ms post-picture onset.

**lmm_cueTask.py**  
Prepares LMM-ready data tables and computes descriptive statistics.  
- Loads pupil data, fixation valences, block order, and dominant-eye info.  
- Extracts pupil size at +500 ms post-picture onset.  
- Builds a DataFrame with participant IDs, current valence, previous valence, task condition, and pupil values.  
- Prints descriptive statistics by condition, valence, previous valence, and their interactions.  
- Fits linear mixed-effects models:
  1. `pupil500 ~ valence * previous_valence + (1|participant)`
  2. `pupil500 ~ valence + previous_valence + condition + (1|participant)`  
  using `pymer4`.
