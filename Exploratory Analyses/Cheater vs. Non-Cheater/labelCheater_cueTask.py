import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm 
import pandas as pd  # For color mapping

# === LOAD DATA ===

dominantEye = sio.loadmat("C:/Users/charl/OneDrive/Dokumente/Uni/7. Semester/Bachelor Thesis/ArrowTask/dominantEye_clean.mat")['dominantEye']
for i in range(48): 
    if dominantEye[0,i] == 0: 
        xPosition  = sio.loadmat("C:/Users/charl/OneDrive/Dokumente/Uni/7. Semester/Bachelor Thesis/ArrowTask/xPositionRightPic_clean.mat")["xPositionRight"]
    elif dominantEye[0,i] == 1: 
        xPosition  = sio.loadmat("C:/Users/charl/OneDrive/Dokumente/Uni/7. Semester/Bachelor Thesis/ArrowTask/xPositionLeftPic_clean.mat")["xPositionLeft"]

timeVector = sio.loadmat("C:/Users/charl/OneDrive/Dokumente/Uni/7. Semester/Bachelor Thesis/ArrowTask/timeVectorPic_clean.mat")["timeVector"]
calibration = sio.loadmat("C:/Users/charl/OneDrive/Dokumente/Uni/7. Semester/Bachelor Thesis/ArrowTask/calibrationShifts.mat")["calibrationShifts"]

num_blocks = 48
num_trials = 88
picOnset_idx = 500

cheater = np.full((num_blocks, num_trials), np.nan)

for b in range(num_blocks): 
    for t in range(num_trials):
        center = calibration[b,t]
        lower = center - 100 
        upper = center + 100 

        if lower <= xPosition[b, t, picOnset_idx] <= upper: 
            cheater[b, t] = 0 # if gaze is within thresholds at pic onset --> non cheater 
        
        else: 
            cheater[b, t] = 1 # else cheater


sio.savemat(
    "C:/Users/charl/OneDrive/Dokumente/Uni/7. Semester/Bachelor Thesis/ArrowTask/cheater.mat",
    {"cheater": cheater})