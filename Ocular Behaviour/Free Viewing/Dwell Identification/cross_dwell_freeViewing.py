"""
Free Viewing — Cross dwell detection (x-position based)

What it does:
  • Loads cross→cross epoched x-position/time data (test trials removed)
  • Loads picture onset times (absolute, from CSV)
  • Detects the longest stable dwell on the fixation-cross band BEFORE picture onset
      - primary band: [fixation_cross_lower, fixation_cross_upper]
      - if not found, estimate a per-trial center and retry with ±100px around it
  • Saves:
      - fixationCrossStartTimes.mat   (ms)
      - fixationCrossEndTimes.mat     (ms)
      - calibrationShifts.mat         (px; estimated center used for fallback)
  • Debug/visualization section to plot one trial with automatic labels
"""

import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import csv

# ==============================
# Parameters
# ==============================
# Fixation detection
stability_threshold = 50                
start_end_stability_threshold = 50       
window_size = 30                         
min_out_of_threshold_duration = 50      

# Fixation cross horizontal band
default_center = 960  
fixation_cross_lower = default_center - 100
fixation_cross_upper = default_center + 100
                    

# Paths 
ROOT = r"C:/Users/charl/OneDrive/Dokumente/Uni/7. Semester/Bachelor Thesis/Eye Data Experiment1Task2"
PATH_CONT = f"{ROOT}/Continuos"
PATH_FIX  = f"{ROOT}/Experiment1_Task2_Fixations"

MAT_EPO_X = f"{PATH_CONT}/epochedXPosition.mat"          
MAT_EPO_T = f"{PATH_CONT}/epochedTimeVector.mat"         
MAT_PIC_FIX_START = f"{PATH_FIX}/fixationPictureStartTimes.mat"  
CSV_PIC_ONSETS = f"{PATH_CONT}/pictureOnsetTimes.csv"   

OUT_FIX_START = f"{PATH_CONT}/fixationCrossStartTimes.mat"
OUT_FIX_END   = f"{PATH_CONT}/fixationCrossEndTimes.mat"
OUT_CAL_SHIFT = f"{PATH_CONT}/calibrationShifts.mat"

# Remove the first 4 test trials everywhere
DROP_TEST_N = 4

# ==============================
# Load epoched data
# ==============================
x_data = sio.loadmat(MAT_EPO_X)['epochedXPosition']      # shape: (participants, trials) object
t_data = sio.loadmat(MAT_EPO_T)['epochedTimeVector']     # shape: (participants, trials) object
pic_fixation_start_times = sio.loadmat(MAT_PIC_FIX_START)['fixationStartTimes']

# Drop first 4 test trials
x_data = x_data[:, DROP_TEST_N:]
t_data = t_data[:, DROP_TEST_N:]
pic_fixation_start_times = pic_fixation_start_times[:, DROP_TEST_N:]
print(f"Loaded epoched data (after dropping first {DROP_TEST_N} trials): {x_data.shape}")

# ==============================
# Load picture onset times (absolute ms, CSV)
# ==============================
pic_onsets = []
with open(CSV_PIC_ONSETS, newline='') as f:
    rdr = csv.reader(f)
    _ = next(rdr)  # header
    for row in rdr:
        pic_onsets.append([float(v) if v else np.nan for v in row])
pic_onsets = np.array(pic_onsets)
pic_onsets = pic_onsets[:, DROP_TEST_N:]  # align with dropped trials
print("Picture onsets (ms) shape:", pic_onsets.shape)

# ==============================
# Output arrays (ms & px)
# ==============================
n_participants, n_trials = x_data.shape
fixation_start_times = np.full((n_participants, n_trials), np.nan)
fixation_end_times   = np.full((n_participants, n_trials), np.nan)
calibration_shifts   = np.full((n_participants, n_trials), np.nan)

# ==============================
# Helper — detect longest stable fixation in a band before a cutoff time
# ==============================
def detect_longest_fixation(x, t, lower, upper, end_time=None):
    """
    x: x-position trace (px)
    t: time vector (ms, absolute)
    lower/upper: band limits (px)
    end_time: stop scanning at this absolute time (e.g., picture onset)
    Returns (start_ms, end_ms) for the longest dwell, or (None, None).
    """
    fixations = []
    fixation_start = None
    consecutive_out = 0

    for i in range(len(x)):
        if end_time is not None and t[i] >= end_time:
            break

        xi = x[i]
        if np.isnan(xi):
            continue

        # Need a full window to check stability at start
        if i + window_size >= len(x):
            break

        window = x[i:i + window_size]

        if fixation_start is None:
            # inside band & stable window to start a dwell
            if lower <= np.nanmin(window) and np.nanmax(window) <= upper:
                if (np.nanmax(window) - np.nanmin(window)) <= start_end_stability_threshold:
                    fixation_start = t[i + 2]  # small offset to avoid window edge
        else:
            if lower <= xi <= upper:
                consecutive_out = 0
            else:
                consecutive_out += 1
                if consecutive_out > min_out_of_threshold_duration:
                    fixation_end = t[i - min_out_of_threshold_duration]
                    fixations.append((fixation_start, fixation_end))
                    fixation_start = None
                    consecutive_out = 0

    # If still in a dwell when we hit end_time, close it
    if fixation_start is not None and end_time is not None:
        fixations.append((fixation_start, end_time))

    if len(fixations) == 0:
        return None, None

    # Longest dwell
    return max(fixations, key=lambda f: f[1] - f[0])

# ==============================
# Main loop — find cross dwells
# ==============================
missing = []

for p in range(n_participants):
    for tr in range(n_trials):
        x = np.array(x_data[p, tr]).flatten()
        t = np.array(t_data[p, tr]).flatten()
        pic_time = float(pic_onsets[p, tr]) if np.isfinite(pic_onsets[p, tr]) else np.nan
        pic_fix_start = float(pic_fixation_start_times[p, tr]) if np.isfinite(pic_fixation_start_times[p, tr]) else np.nan

        if np.isnan(pic_time) or len(t) == 0:
            # No picture time or empty epoch → skip
            missing.append((p, tr))
            continue

        # 1) Try with the global cross band
        s_ms, e_ms = detect_longest_fixation(x, t, fixation_cross_lower, fixation_cross_upper, end_time=pic_time)
        est_center = default_center

        # 2) Fallback: estimate per-trial center from first stable period before picture
        if s_ms is None:
            # Find a stable segment BEFORE picture onset
            last_idx = np.searchsorted(t, pic_time) - window_size - 1
            if last_idx > 200:
                found = False
                for i in range(200, last_idx):
                    w = x[i:i + window_size]
                    if np.all(np.isfinite(w)) and (np.nanmax(w) - np.nanmin(w)) <= stability_threshold:
                        # take median to estimate center
                        start_i = i + 2
                        end_i = min(start_i + 300, len(x))
                        est_center = np.nanmedian(x[start_i:end_i])
                        found = True
                        break
                if found:
                    s_ms, e_ms = detect_longest_fixation(x, t, est_center - 100, est_center + 100, end_time=pic_time)

        if s_ms is not None and e_ms is not None:
            fixation_start_times[p, tr] = s_ms
            fixation_end_times[p, tr]   = e_ms
            calibration_shifts[p, tr]   = est_center
        else:
            missing.append((p, tr))

print(f"Done. Missing/undetected dwells: {len(missing)}")

# ==============================
# Save results
# ==============================
sio.savemat(OUT_FIX_START, {"fixationStartTimes": fixation_start_times})
sio.savemat(OUT_FIX_END,   {"fixationEndTimes":   fixation_end_times})
sio.savemat(OUT_CAL_SHIFT, {"calibrationShifts":  calibration_shifts})
print("Saved:",
      "\n  ", OUT_FIX_START,
      "\n  ", OUT_FIX_END,
      "\n  ", OUT_CAL_SHIFT)

# =============================================================================
#                                PLOT
# =============================================================================

TRIALS_PER_BLOCK = 88

participant_number = 10
trial_number = 55
p_idx = participant_number - 1
tr_idx = trial_number - 1

x = np.array(x_data[p_idx, tr_idx]).flatten()
t = np.array(t_data[p_idx, tr_idx]).flatten()
fix_start = fixation_start_times[p_idx, tr_idx]
fix_end   = fixation_end_times[p_idx, tr_idx]
pic_time  = pic_onsets[p_idx, tr_idx]
est_center_this = calibration_shifts[p_idx, tr_idx]

default_center = 960.0
default_lower, default_upper = default_center - 100, default_center + 100

# used thresholds = adapted if est_center_this is finite, else default
if np.isfinite(est_center_this):
    used_center = est_center_this
else:
    used_center = default_center
used_lower, used_upper = used_center - 100, used_center + 100

used_adapted = abs(used_center - default_center) > 1e-6

t0 = t[0] if t.size > 0 else 0.0
t_rel = t - t0
pic_rel = pic_time - t0 if np.isfinite(pic_time) else np.nan
fix_start_rel = fix_start - t0 if np.isfinite(fix_start) else np.nan
fix_end_rel   = fix_end   - t0 if np.isfinite(fix_end)   else np.nan

block_number = (tr_idx // TRIALS_PER_BLOCK) + 1

plt.figure(figsize=(10, 5))
plt.plot(t_rel, x, color='black', lw=1.0, label="X position")
plt.axvline(0, color='green', ls='--', label="Cross onset")
if np.isfinite(pic_rel):
    plt.axvline(pic_rel, color='red', ls='--', label="Picture onset")
if np.isfinite(fix_start_rel) and np.isfinite(fix_end_rel):
    in_fix = (t_rel >= fix_start_rel) & (t_rel <= fix_end_rel)
    if np.any(in_fix):
        plt.plot(t_rel[in_fix], x[in_fix], color='red', lw=2.0, label="Dwell on cross")

# plot the thresholds that were actually used 
plt.axhline(used_lower, color='tab:blue', ls='-.', label="Lower threshold")
plt.axhline(used_upper, color='tab:blue', ls='-.', label="Upper threshold")

# if adapted, also show the default band 
if used_adapted:
    plt.axhline(default_lower, color='orange', ls=':', lw=1.8, label="Original Lower Threshold")
    plt.axhline(default_upper, color='orange', ls=':', lw=1.8, label="Original Upper Threshold")

plt.xlabel("Time (ms) relative to cross onset")
plt.ylabel("X Position")
plt.title(f"X Position (Participant {participant_number}, Block {block_number}, Trial {trial_number})")
plt.legend(loc="best")
plt.tight_layout()
plt.show()
