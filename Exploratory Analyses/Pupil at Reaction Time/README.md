# Exploratory Analyses — Pupil at Reaction Time (Cue Task)

This folder contains scripts for analyzing pupil responses time-locked to participants' reaction times (RTs) in the Cue Task. Analyses include both visualization of RT-locked pupil traces and linear mixed model (LMM) statistics at the RT.

## Files in this folder

**plotPupilRT_cueTask.py**  
Plots average pupil size aligned to reaction time for congruent and incongruent trials.  
- Loads RT-locked pupil data (left/right eyes), time vector, block order, and dominant-eye information.  
- Determines task condition per trial based on block order and trial number  
- Extracts dominant-eye pupil traces.  
- Aggregates trials by task condition.  
- Computes mean pupil size and 95% confidence intervals across participants.  
- Produces a time-course plot with 0 ms aligned to the reaction time.

**lmmPupilRT_cueTask.py**  
Performs participant-level and LMM analyses of pupil size at RT.  
- Loads RT-locked pupil data, first and last fixation valences, block order, and dominant-eye information.  
- Extracts the pupil size at the reaction time index (index 500).  
- Assigns trial labels: current valence, previous valence, and task condition (congruent/incongruent).  
- Builds a pandas DataFrame for statistical analysis.  
- Computes descriptive statistics (mean, SD, min, max) for pupil size at RT per task condition.  
- Runs an LMM with pupil size at RT as the dependent variable: pupilRT ~ valence + previous_valence + condition + (1|participant)
