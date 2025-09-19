# Data Preparation — Free Viewing

This folder contains scripts for cleaning, epoching, and checking pupil & gaze data for the Free Viewing task.

---

## Files in this folder

### `pupil_preprocessing_freeViewing.py`
- Cleans continuous pupil traces (mask invalids, remove jumps, interpolate, median-filter).  
- Saves cleaned left/right pupil data as `.mat` files.  
- Includes Quality Check plot (before vs. after cleaning).  

### `epoching_freeViewing.py`
- Segments continuous signals in two different ways:  
  - Cross onset → Cross onset (variable length)  
  - around Picture onset (fixed length)  
- Saves pupil, gaze, and time vectors in mat files.  

### `PupilDataCheck_freeViewing.py` 
- Plots single trials
- plots grand average with CI
- plots all pupil traces for positive and for negative trials
- Includes plot of pupil size value distributions per participant
