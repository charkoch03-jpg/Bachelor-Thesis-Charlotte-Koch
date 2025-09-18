"""
Pupil preprocessing for Free Viewing task

Steps:
1. Mask invalid values (0 and 32700) ±5 samples
2. Remove extreme fluctuations (derivative threshold ±2 SD)
3. Interpolate missing samples linearly
4. Median filter (kernel size = 11)
5. Plot one example trace before/after preprocessing
6. Save cleaned signals as .mat files
"""

import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from scipy.ndimage import binary_dilation
import pandas as pd
from scipy.signal import medfilt

# === Load continuous pupil size data (MATLAB cell arrays) ===
PupilSizeRight = sio.loadmat("C:/Users/charl/OneDrive/Dokumente/Uni/7. Semester/Bachelor Thesis/Eye Data Experiment1Task2/Continuos/pupilSizeRightContinuos.mat")['pupilSizeRight']
PupilSizeLeft = sio.loadmat("C:/Users/charl/OneDrive/Dokumente/Uni/7. Semester/Bachelor Thesis/Eye Data Experiment1Task2/Continuos/pupilSizeLeftContinuos.mat")['pupilSizeLeft']

# Flatten and combine all signals for quick global QC
all_left = np.concatenate([entry[0].flatten() for entry in PupilSizeLeft])
all_right = np.concatenate([entry[0].flatten() for entry in PupilSizeRight])

print("\nInitial Data Quality Check:")
print("Left eye - zeros:", np.sum(all_left == 0))
print("Left eye - 32700:", np.sum(all_left == 32700))
print("Left eye - NaN:", np.sum(np.isnan(all_left)))
print("Right eye - zeros:", np.sum(all_right == 0))
print("Right eye - 32700:", np.sum(all_right == 32700))
print("Right eye - NaN:", np.sum(np.isnan(all_right)))

# === Example raw trace ===
plt.figure(figsize=(12, 4))
plt.plot(PupilSizeLeft[0][0], label="Raw Left Eye (Participant 1)", alpha=0.7)
plt.title("Raw Left Eye Pupil Signal (Before Preprocessing)")
plt.xlabel("Sample Index")
plt.ylabel("Pupil Size (a.u.)")
plt.legend()
plt.tight_layout()
plt.show()

# === Step 1: Mask invalid values (0 / 32700 ± 5 samples) ===
def mask_invalid(data, window=5):
    cleaned = []
    struct = np.ones(2 * window + 1)
    for entry in data:
        sig = entry[0].astype(float).flatten()
        bad = (sig == 0) | (sig == 32700)
        bad_expanded = binary_dilation(bad, structure=struct)
        sig[bad_expanded] = np.nan
        cleaned.append(sig)
    return cleaned

left_masked = mask_invalid(PupilSizeLeft)
right_masked = mask_invalid(PupilSizeRight)

# Plot before/after masking
plt.figure(figsize=(12, 4))
plt.plot(PupilSizeLeft[0][0], label="Original", alpha=0.3)
plt.plot(left_masked[0], label="Masked invalids", alpha=0.8)
plt.title("Masking invalid values (Participant 1, Left Eye)")
plt.xlabel("Sample Index")
plt.ylabel("Pupil Size (a.u.)")
plt.legend()
plt.tight_layout()
plt.show()

# === Step 2: Remove extreme fluctuations ===
def remove_fluctuations(data, z_thresh=2, window=5):
    diffs_all = np.concatenate([np.diff(x) for x in data if len(x) > 1])
    mean_diff = np.nanmean(diffs_all)
    std_diff = np.nanstd(diffs_all)
    lower, upper = mean_diff - z_thresh * std_diff, mean_diff + z_thresh * std_diff

    cleaned = []
    for sig in data:
        s = sig.copy()
        diffs = np.diff(s)
        bad_idx = np.where((diffs < lower) | (diffs > upper))[0]
        for idx in bad_idx:
            start = max(0, idx - window)
            end = min(len(s), idx + 1 + window)
            s[start:end] = np.nan
        cleaned.append(s)
    return cleaned

left_clean = remove_fluctuations(left_masked)
right_clean = remove_fluctuations(right_masked)

# === Step 3: Linear interpolation ===
def interpolate_nan(data):
    out = []
    for sig in data:
        series = pd.Series(sig)
        interp = series.interpolate(method="linear", limit_direction="both").to_numpy()
        out.append(interp)
    return out

left_interp = interpolate_nan(left_clean)
right_interp = interpolate_nan(right_clean)

# === Step 4: Median filter ===
def median_filter(data, kernel=11):
    out = []
    for sig in data:
        k = min(kernel, len(sig) // 2 * 2 + 1)  # ensure odd kernel <= length
        out.append(medfilt(sig, kernel_size=k))
    return out

left_final = median_filter(left_interp, kernel=11)
right_final = median_filter(right_interp, kernel=11)

# === Final before/after plot (example trace) ===
plt.figure(figsize=(12, 4))
plt.plot(PupilSizeLeft[0][0], label="Before preprocessing", alpha=0.6)
plt.plot(left_final[0], label="After preprocessing", alpha=0.9)
plt.title("Participant 1 Block 1 (Free Viewing Task) - Left Eye")
plt.xlabel("Sample Index")
plt.ylabel("Pupil Size")
plt.legend()
plt.tight_layout()
plt.show()

# === Save cleaned data ===
sio.savemat(
    "data/interim/free_viewing/pupilSizeLeft_cleanedInterpolated.mat",
    {"pupilSizeLeft": np.array([x.reshape(-1, 1) for x in left_final], dtype=object).reshape(-1, 1)},
)
sio.savemat(
    "data/interim/free_viewing/pupilSizeRight_cleanedInterpolated.mat",
    {"pupilSizeRight": np.array([x.reshape(-1, 1) for x in right_final], dtype=object).reshape(-1, 1)},
)

print("\nPreprocessing complete. Cleaned data saved.")

