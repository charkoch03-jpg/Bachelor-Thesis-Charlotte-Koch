import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.io as sio

pupilSizeRight = sio.loadmat("C:/Users/charl/OneDrive/Dokumente/Uni/7. Semester/Bachelor Thesis/ArrowTask/pupilSizeRight_epoched_RTLocked.mat")['pupilSizeRight']
pupilSizeLeft = sio.loadmat("C:/Users/charl/OneDrive/Dokumente/Uni/7. Semester/Bachelor Thesis/ArrowTask/pupilSizeLeft_epoched_RTLocked.mat")['pupilSizeLeft']
timeVector = sio.loadmat("C:/Users/charl/OneDrive/Dokumente/Uni/7. Semester/Bachelor Thesis/ArrowTask/timeVector_epoched_RTLocked.mat")['timeVector']
dominantEye = sio.loadmat("C:/Users/charl/OneDrive/Dokumente/Uni/7. Semester/Bachelor Thesis/ArrowTask/dominantEye_clean.mat")["dominantEye"]   

import numpy as np
import pandas as pd
import scipy.io as sio
import matplotlib.pyplot as plt

# ==============================
# Load data (RT-locked)
# ==============================
base = r"C:/Users/charl/OneDrive/Dokumente/Uni/7. Semester/Bachelor Thesis/ArrowTask/"

pupilRight = sio.loadmat(base + "pupilSizeRight_epoched_RTLocked.mat")["pupilSizeRight"]  
pupilLeft  = sio.loadmat(base + "pupilSizeLeft_epoched_RTLocked.mat")["pupilSizeLeft"]    
timeVector = sio.loadmat(base + "timeVector_epoched_RTLocked.mat")["timeVector"]          
dominantEye = sio.loadmat(base + "dominantEye_clean.mat")["dominantEye"]                  
blockOrder  = sio.loadmat(base + "blockOrder_clean.mat")["blockOrder"]                    

# ==============================
# shapes and time axis
# ==============================
B, T, S = pupilRight.shape  

time_ms = timeVector[0, 0, :].astype(float) if timeVector.ndim == 3 else np.squeeze(timeVector).astype(float)

# ==============================
# Collect trials by task condition
# If block order == 'A' -> first half congruent, second half incongruent; if 'B' -> reversed
# ==============================
half = T // 2  

cong_trials = []  
incg_trials = []  

for b in range(B):
    # pick data based on dominant eye
    dom = int(dominantEye[0, b]) if dominantEye.ndim == 2 else int(dominantEye[b])
    block_pupil = pupilLeft[b] if dom == 0 else pupilRight[b]   

    # block order ('A' or 'B')
    order = str(blockOrder[0, b]) if blockOrder.ndim == 2 else str(blockOrder[b])

    for t in range(T):
        # condition for this trial
        if order == 'A':
            cond = 'congruent' if t < half else 'incongruent'
        else:
            cond = 'incongruent' if t < half else 'congruent'

        x = block_pupil[t, :].astype(float)

        # store
        if cond == 'congruent':
            cong_trials.append(x)
        else:
            incg_trials.append(x)

# Convert lists to arrays
cong_arr = np.vstack(cong_trials) if len(cong_trials) > 0 else np.empty((0, S))
incg_arr = np.vstack(incg_trials) if len(incg_trials) > 0 else np.empty((0, S))

print(f"Collected {cong_arr.shape[0]} congruent trials, {incg_arr.shape[0]} incongruent trials.")

# ==============================
# Compute mean and simple 95% CI across trials
# ==============================
def mean_ci(arr, conf=0.95):
    m = np.nanmean(arr, axis=0)
    sd = np.nanstd(arr, axis=0, ddof=1)
    n  = 24
    sem = np.where(n > 1, sd / np.sqrt(n), np.nan)
    z = 1.96
    lo = m - z * sem
    hi = m + z * sem
    return m, lo, hi

m_cg, lo_cg, hi_cg = mean_ci(cong_arr) if cong_arr.size else (np.full(S, np.nan),)*3
m_ic, lo_ic, hi_ic = mean_ci(incg_arr) if incg_arr.size else (np.full(S, np.nan),)*3

# ==============================
# Plot (RT-locked; 0 ms = reaction time)
# ==============================
plt.figure(figsize=(10, 5))
plt.plot(time_ms, m_cg, label="Congruent", linewidth=1.8, color='tab:blue')
plt.fill_between(time_ms, lo_cg, hi_cg, alpha=0.2)

plt.plot(time_ms, m_ic, label="Incongruent", linewidth=1.8, color='tab:orange')
plt.fill_between(time_ms, lo_ic, hi_ic, alpha=0.2)

plt.axvline(0, linestyle="--", linewidth=1)  # RT at 0 ms
plt.xlabel("Time (ms) relative to RT")
plt.ylabel("Pupil size (A.U.)")
plt.title("Average Pupil Size over Time for Task Condition")
plt.legend()
plt.grid(alpha=0.3, linestyle="--")
plt.tight_layout()
plt.show()
