# Dwell Identification — Free Viewing

This folder contains scripts for detecting and summarizing cross and picture dwells based on x-position data.

## Files in this folder

**cross_dwell_freeViewing.py**  
Detects the longest stable dwell on the fixation cross before picture onset.  
- Loads cross→cross epoched x-position/time data and picture onset times, uses dominant eye's data  
- Detects dwells inside a horizontal band (default ±100 px), with fallback to per-trial estimated center.  
- Saves `fixationCrossStartTimes.mat`, `fixationCrossEndTimes.mat`, `calibrationShifts.mat`.  
- Includes a plot for single-trial dwell visualization.  

**crossDwellStats_freeViewing.py**  
Prints and plots descriptive statistics of cross dwells.  
- Computes dwell start, end, and duration relative to cross onset.  
- plots distributions  

**picture_dwell_freeViewing.py**  
Detects first and last picture dwells per trial.  
- Uses dominant-eye x-position data.  
- Detects stable dwell windows based on velocity and stability thresholds.  
- Computes dwell sides and valence.  
- Saves `fixationPictureStartTimes.mat`, `fixationPictureEndTimes.mat`, `firstFixationSide.mat`, `lastFixationSide.mat`, `lastFixationPictureStartTimes.mat`, `lastFixationPictureEndTimes.mat`, `lastFixationValence.mat`.  

**pictureDwellStats_freeViewing.py**  
Prints and plots descriptive statistics of first and last picture dwells.  
- Converts absolute dwell times to values relative to picture onset.  
- Computes and visualizes start, end, and duration distributions of first and last picture dwell
