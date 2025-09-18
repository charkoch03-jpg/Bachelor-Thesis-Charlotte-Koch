# Gaze Behaviour — Free Viewing

This folder contains scripts for visualizing gaze behavior during free viewing, including first/last dwell sequences and gaze over time.

## Files in this folder

**GazeBarPlot_freeViewing.py**  
Plots categorical counts of gaze sequences across trials.  
- Compares first vs. last picture dwell side: same vs. different.  
- Further splits into three categories: First only, First → Second, First → Second → First.  
- Produces bar plots with trial counts.

**GazeTimePlot_freeViewing.py**  
Plots continuous gaze behavior over time relative to picture onset.  
- Uses dominant-eye-aware x-position data and calibration shifts.  
- Computes percentage of trials looking at cross (center), first picture side, second picture side, and returning to first.  
- Produces stacked area plot over time.
