# Data Preparation — Cue Task

This folder contains scripts for cleaning, epoching, and checking pupil & gaze data, as well as preprocessing and summarizing behavioral data for the Cue/Arrow task.

---

## Files in this folder

### `pupil_preprocessing_cueTask.py`
- Cleans continuous pupil traces (mask invalids, remove jumps, interpolate, median-filter).  
- Saves cleaned left/right pupil data as `.mat` files.  
- Includes Quality Check plot (before vs. after cleaning).  

### `epoching_cueTask.py`
- Segments continuous signals in three different ways:  
  - Cross onset → Cross onset (variable-length)  
  - around Picture onset (fixed length)  
  - around Reaction Time (fixed length)  
- Saves pupil, x position, and time vectors in mat files. 

### `pupilDataCheck_cueTask.py`  
- Plots single trials
- Plots grand averages with CI
- Plots all pupil traces for positive and negative trials separately  

### `behaviouralPreprocessing_cueTask.py`
- Removes test trials and Participants to be removed
- Removes trials with incorrect answers or too fast reaction times  
- Computes accuracy per participant, winsorizes RTs, and saves log-transformed RTs.  
- Outputs cleaned behavioral and eyetracking files.  

### `behaviouralStats_cueTask.py`
- Prints descriptive stats of cleaned RTs (N, mean, SD, min, max).  


