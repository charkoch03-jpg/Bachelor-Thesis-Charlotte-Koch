# Dwell Identification — Free Viewing

This folder contains scripts for detecting and summarizing cross and picture dwells from epoched x-position and time vector data.

## Files in this folder

**cross_dwell_freeViewing.py**  
Detects the longest stable dwell on the fixation cross before picture onset.  
- Loads cross→cross epoched x-position/time data and picture onset times.  
- Detects dwells inside a horizontal band (default ±100 px), with fallback to per-trial estimated center.  
- Saves `fixationCrossStartTimes.mat`, `fixationCrossEndTimes.mat`, `calibrationShifts.mat`.  
- Includes a debug plot for single-trial dwell visualization.  

**crossDwellStats_freeViewing.py**  
Prints and plots descriptive statistics of cross dwells.  
- Computes dwell start, end, and duration relative to cross onset.  
- Produces histograms with adaptive binning.  

**picture_dwell_freeViewing.py**  
Detects first and last picture dwells per trial.  
- Uses dominant-eye-aware x-position data.  
- Detects stable dwell windows based on velocity and stability thresholds.  
- Computes dwell side and last-dwell valence.  
- Saves `fixationPictureStartTimes.mat`, `fixationPictureEndTimes.mat`, `firstFixationSide.mat`, `lastFixationSide.mat`, `lastFixationPictureStartTimes.mat`, `lastFixationPictureEndTimes.mat`, `lastFixationValence.mat`.  

**pictureDwellStats_freeViewing.py**  
Prints and plots descriptive statistics of first and last picture dwells.  
- Converts absolute dwell times to values relative to picture onset.  
- Computes and visualizes start, end, and duration distributions.
