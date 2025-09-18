# Data Preparation — Cue Task

This folder contains scripts for cleaning, epoching, and checking pupil & gaze data, as well as preprocessing and summarizing behavioral data for the Cue/Arrow task.

---

## Files in this folder

### `pupil_preprocessing_cueTask.py`
- Cleans continuous pupil traces (mask invalids, remove jumps, interpolate, median-filter).  
- Saves cleaned left/right pupil data as `.mat` files.  
- Includes QC plot (before vs. after cleaning).  

### `epoching_cueTask.py`
- Segments continuous signals into three modes:  
  - Cross→Cross (variable-length)  
  - Picture-locked (fixed window)  
  - RT-locked (±1000 ms around RT)  
- Saves pupil, gaze, and time vectors in MATLAB-compatible format.  

### `pupilDataCheck_cueTask.py`
- Quick visual QA of picture-locked pupil signals.  
- Plots single trials, grand averages with CI, and condition overlays.  
- Uses dominant eye and trial labels (cue side, picture valence).  

### `behaviouralPreprocessing_cueTask.py`
- Cleans and aligns behavioral variables with pupil data.  
- Drops practice/test trials, invalidates incorrect/too-fast responses.  
- Computes summary accuracy, winsorizes RTs, and saves log-transformed RTs.  
- Outputs cleaned behavioral and pupil-epoched files.  

### `behaviouralStats_cueTask.py`
- Prints descriptive stats of cleaned RTs (N, mean, SD, min, max).  


