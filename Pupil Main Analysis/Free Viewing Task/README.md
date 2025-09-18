# Pupil Main Analysis — Free Viewing

This folder contains scripts for analyzing pupil responses during free viewing, including plotting time courses and preparing data for linear mixed models (LMM).

## Files in this folder

**PupilPlots_freeViewing.py**  
Prepares and visualizes pupil size data across trials and participants.  
- Loads left/right eye pupil traces and dominant-eye information.  
- Removes invalid participants and test trials.  
- Splits data by current valence, previous valence, and their interaction.  
- Computes participant-level means and 95% CI.  
- Produces time-course plots of pupil responses for each group.  
- Prints descriptive statistics at 500 ms post-picture onset.

**lmm_freeViewing.py**  
Prepares LMM-ready data tables and calculates descriptive stats.  
- Loads epoched pupil data, dominant-eye info, valence, and last fixation valence.  
- Excludes invalid participants and test trials.  
- Extracts pupil size at 500 ms post-picture onset.  
- Constructs a DataFrame with participant IDs, current valence, previous valence, and pupil values.  
- Prints descriptive statistics by valence, previous valence, and their interaction.  
- Fits a linear mixed-effects model (`pupil500 ~ valence * previous_valence + (1|participant)`) using `pymer4`.
