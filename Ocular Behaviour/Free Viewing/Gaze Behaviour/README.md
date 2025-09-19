# Gaze Behaviour — Free Viewing

This folder contains scripts for visualizing gaze behavior during the free viewing task, including first/last dwell sequences and gaze behaviour over time.

## Files in this folder

**GazeBarPlot_freeViewing.py**  
Plots categorical counts of gaze sequences across trials.  
- splits into three categories: First only, First → Second, First → Second → First.  
- Produces bar plots with trial counts belonging to each category

**GazeTimePlot_freeViewing.py**  
Plots continuous gaze behavior over time relative to picture onset.  
- Uses dominant-eye x-position data  
- Computes percentage of trials looking at cross (center), first picture side, second picture side, and returning to first.  
- Produces stacked area plot over time.
