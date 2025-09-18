import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from scipy.stats import t

# ==============================
# Paths & load
# ==============================
pupilSizeLeft = sio.loadmat(
    "C:/Users/charl/OneDrive/Dokumente/Uni/7. Semester/Bachelor Thesis/ArrowTask/pupilSizeLeftPic_clean.mat"
)["pupilSizeLeft"]

timeVector = sio.loadmat(
    "C:/Users/charl/OneDrive/Dokumente/Uni/7. Semester/Bachelor Thesis/ArrowTask/timeVectorPic_clean.mat"
)['timeVector']

side = sio.loadmat(
    "C:/Users/charl/OneDrive/Dokumente/Uni/7. Semester/Bachelor Thesis/ArrowTask/firstFixationSide.mat"
)['firstFixationSide']

n_blocks, n_trials, n_timepoints = pupilSizeLeft.shape

# ==============================
# Separate trials by first looked side
# ==============================
mask_side0 = (side == 0)
mask_side1 = (side == 1)

pupil_side0 = []
pupil_side1 = []

for b in range(n_blocks):
    for t in range(n_trials):
        if mask_side0[t, b]:
            pupil_side0.append(pupilSizeLeft[b, t, :])
        elif mask_side1[t, b]:
            pupil_side1.append(pupilSizeLeft[b, t, :])

# Convert lists to numpy arrays
pupil_side0 = np.array(pupil_side0)
pupil_side1 = np.array(pupil_side1)

n_participants = 24

# Compute mean and SEM
mean_side0 = np.nanmean(pupil_side0, axis=0)
std_side0 = np.nanstd(pupil_side0, axis=0, ddof=1)
sem_side0 = std_side0 / np.sqrt(n_participants)

mean_side1 = np.nanmean(pupil_side1, axis=0)
std_side1 = np.nanstd(pupil_side1, axis=0, ddof=1)
sem_side1 = std_side1 / np.sqrt(n_participants)

# ==============================
# Use MATLAB time vector
# ==============================
time_ms = timeVector[0,0,:] - timeVector[0,0,500]  # picture onset at 0 ms
time_ms = time_ms.flatten()  # make 1D for plotting

# ==============================
# Plot
# ==============================
plt.figure(figsize=(10, 5))
plt.plot(time_ms, mean_side0, label="Cue Side = Left")
plt.fill_between(time_ms, mean_side0 - sem_side0, mean_side0 + sem_side0, alpha=0.3)

plt.plot(time_ms, mean_side1, label="Cue Side = Right")
plt.fill_between(time_ms, mean_side1 - sem_side1, mean_side1 + sem_side1, alpha=0.3)

plt.axvline(x=0, color='k', linestyle='--', label='Picture Onset')
plt.axvline(x=500, color='blue', linestyle='--', label='500ms post Picture Onset')
plt.xlabel("Time (ms) relative to Picture Onset")
plt.ylabel("Average Pupil Size")
plt.title("Average Pupil Size Over Time by Cue Side")
plt.legend()
plt.tight_layout()
plt.show()
