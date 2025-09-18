# Data Preparation — Free Viewing

This folder contains scripts for cleaning, epoching, and checking pupil & gaze data for the Free Viewing task.

---

## Files in this folder

### `pupil_preprocessing_freeViewing.py`
- Cleans continuous pupil traces (mask invalids, remove jumps, interpolate, median-filter).  
- Saves cleaned left/right pupil data as `.mat` files.  
- Includes QC plot (before vs. after cleaning).  

### `epoching_freeViewing.py`
- Segments continuous signals into two modes:  
  - Cross→Cross (variable-length)  
  - Picture-locked (fixed window)  
- Saves pupil, gaze, and time vectors in MATLAB-compatible format.  

### `PupilDataCheck_freeViewing.py` 
- Plots single trials, grand averages with CI, and overlays per condition.  
- Includes per-participant distribution plots (dominant eye).  
