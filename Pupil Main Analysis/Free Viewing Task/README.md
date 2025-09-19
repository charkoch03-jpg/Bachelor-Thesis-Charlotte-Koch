# Pupil Main Analysis — Free Viewing

This folder contains scripts for analyzing pupil responses during free viewing, including plotting time courses and running linear mixed models (LMM).

## Files in this folder

**PupilPlots_freeViewing.py**  
Prepares and visualizes pupil size data across trials and participants.  
- uses left/right eye pupil traces based on dominant eye.  
- Removes invalid participants and test trials.  
- Splits data by current valence, previous valence, and their interaction.  
- Computes participant-level means and 95% CI.  
- Produces time-course plots of pupil responses for each group (valence, previous valence, interaction)  
- Prints descriptive statistics of pupil size values at 500 ms post-picture onset.

**lmm_freeViewing.py**  
Prepares LMM-ready data tables and calculates descriptive stats.  
- Loads pupil data, dominant-eye info, valence, and last fixation valence.  
- Excludes invalid participants and test trials.   
- Constructs a DataFrame with participant IDs, current valence (first valence), previous valence (last fixation valence from previous trial), and pupil values at 500 ms post-picture onset.  
- Prints descriptive statistics by valence, previous valence, and their interaction.  
- Fits a linear mixed-effects model (`pupil500 ~ valence * previous_valence + (1|participant)`) using `pymer4`.
