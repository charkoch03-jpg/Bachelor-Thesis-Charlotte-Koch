"""
Free Viewing — Pupil size quick visuals

Includes:
  • Plot a single trial (raw trace, onset @ 0 ms)
  • Grand average over time (participant-mean, 95% CI)
  • Overlay all traces per condition (Positive / Negative)
  • Distribution of pupil values per participant (violin)
"""

import numpy as np
import pandas as pd
import scipy.io as sio
import matplotlib.pyplot as plt
from scipy.stats import t
import matplotlib.cm as cm
import seaborn as sns

# ==============================
# Paths & load
# ==============================
ROOT = r"C:/Users/charl/OneDrive/Dokumente/Uni/7. Semester/Bachelor Thesis/Eye Data Experiment1Task2/Experiment1_Task2"

dominantEye = pd.read_csv(f"{ROOT}/dominantEye.csv", header=None).values.flatten()
valenceLeft  = pd.read_csv(f"{ROOT}/valenceLeftEye.csv",  header=None).values
valenceRight = pd.read_csv(f"{ROOT}/valenceRightEye.csv", header=None).values

pupilSizeRight = sio.loadmat(f"{ROOT}/PicturePupilRight_epoched.mat")["PupilSizeRight"]  
pupilSizeLeft  = sio.loadmat(f"{ROOT}/PicturePupilLeft_epoched.mat")["PupilSizeLeft"]
timeVector     = sio.loadmat(f"{ROOT}/PictureTimeVector_epoched.mat")["timeVector"]

# ==============================
# Remove excluded + drop test trials
# ==============================
participants_to_remove = [42, 43, 120, 121]
dominantEye = np.delete(dominantEye, participants_to_remove, axis=0)

pupilSizeLeft  = pupilSizeLeft[:,  4:, :]
pupilSizeRight = pupilSizeRight[:, 4:, :]
timeVector     = timeVector[:,     4:, :]

# ==============================
# Build trial sets (Positive/Negative) using dominant eye + valence
# ==============================
positive_trials, negative_trials = [], []
positive_participant_ids, negative_participant_ids = [], []
skipped_due_to_mismatch, skipped_due_to_too_short = 0, 0

n_blocks = pupilSizeLeft.shape[0]        
n_subjects = n_blocks // 2
TRIALS_PER_BLOCK = pupilSizeLeft.shape[1]
picture_onset_index = 500

for block_idx in range(n_blocks):
    subj_idx = block_idx // 2
    dom_eye  = dominantEye[block_idx]
    val_start, val_end = (0, 88) if (block_idx % 2 == 0) else (88, 176)

    val_block = valenceLeft[subj_idx, val_start:val_end] if dom_eye == 0 else valenceRight[subj_idx, val_start:val_end]
    pupil_blk = pupilSizeLeft[block_idx] if dom_eye == 0 else pupilSizeRight[block_idx]
    time_blk  = timeVector[block_idx]

    for tr in range(TRIALS_PER_BLOCK):
        pup = np.asarray(pupil_blk[tr]).astype(float)
        tv  = np.asarray(time_blk[tr]).astype(float)

        if pup.shape[0] != tv.shape[0]:
            skipped_due_to_mismatch += 1
            continue
        if pup.size < (picture_onset_index + 1):
            skipped_due_to_too_short += 1
            continue

        if val_block[tr] == 1:
            positive_trials.append(pup)
            positive_participant_ids.append(subj_idx)
        elif val_block[tr] == 0:
            negative_trials.append(pup)
            negative_participant_ids.append(subj_idx)

positive_trials = np.array(positive_trials, dtype=float)
negative_trials = np.array(negative_trials, dtype=float)
positive_participant_ids = np.array(positive_participant_ids, dtype=int)
negative_participant_ids = np.array(negative_participant_ids, dtype=int)

# Common timebase (zeroed at picture onset)
time_vec = np.asarray(timeVector[0, 0]).astype(float)
time_zeroed = time_vec - time_vec[picture_onset_index]

# ==============================
# Helpers
# ==============================
def mean_sem_grouped_by_participant(data, participant_ids, confidence=0.95):
    """Participant-level means → grand mean + CI."""
    uniq = np.unique(participant_ids)
    pm = []
    for pid in uniq:
        trials = data[participant_ids == pid]
        if trials.shape[0] > 0:
            pm.append(np.nanmean(trials, axis=0))
    pm = np.array(pm, float)
    mean = np.nanmean(pm, axis=0)
    sem  = np.nanstd(pm, axis=0, ddof=1) / np.sqrt(pm.shape[0]/2)
    tcrit = t.ppf((1 + confidence) / 2, df=max(pm.shape[0] - 1, 1))
    return mean, sem, mean - tcrit * sem, mean + tcrit * sem

def get_trial(subj_idx, block, trial_idx):
    """Return (t_zeroed, pupil) for participant/block/trial."""
    blk = subj_idx * 2 + block
    dom = dominantEye[blk]
    pup_blk = pupilSizeLeft[blk] if dom == 0 else pupilSizeRight[blk]
    tim_blk = timeVector[blk]

    pup = np.asarray(pup_blk[trial_idx]).astype(float)
    tim = np.asarray(tim_blk[trial_idx]).astype(float)
    pup[(pup == 0) | (pup == 32700)] = np.nan
    return tim - tim[picture_onset_index], pup

# ==============================
# 1) Plot a single raw trial
# ==============================
def plot_single_trial(subj_idx=12, block=0, trial_idx=12, title_prefix="Pupil Size over Time"):
    t, y = get_trial(subj_idx, block, trial_idx)
    plt.figure(figsize=(10, 4.6))
    plt.plot(t, y, lw=1.2)
    plt.axvline(0,   color='black', ls='--', label='Picture onset')
    plt.axvline(500, color='blue',  ls='--', label='LMM timepoint')
    plt.xlabel("Time (ms) from picture onset")
    plt.ylabel("Pupil size (A.U.)")
    plt.title(f"{title_prefix} — Participant {subj_idx+1}, Block {block+1}, Trial {trial_idx+1}")
    plt.legend()
    plt.tight_layout()
    plt.show()

# Example calls
plot_single_trial(12, 0, 12)
plot_single_trial(23, 1, 3)

# ==============================
# 2) Grand average over all trials (participant-mean) + 95% CI
# ==============================
all_trials = np.vstack([positive_trials, negative_trials])
all_ids    = np.concatenate([positive_participant_ids, negative_participant_ids])
g_mean, g_sem, g_lo, g_hi = mean_sem_grouped_by_participant(all_trials, all_ids, confidence=0.95)

plt.figure(figsize=(10, 5))
plt.plot(time_zeroed, g_mean, lw=2, label="Mean pupil size")
plt.fill_between(time_zeroed, g_lo, g_hi, alpha=0.2, label="95% CI")
plt.axvline(0,   color='black', ls='--', label='Picture onset')
plt.axvline(500, color='blue',  ls='--', label='LMM timepoint')
plt.xlabel("Time (ms) from picture onset")
plt.ylabel("Pupil size (A.U.)")
plt.title("Average Pupil Size over Time")
plt.legend()
plt.tight_layout()
plt.show()

# ==============================
# 3) Overlay all traces per condition (variance view)
# ==============================
def plot_all_traces(trials, title, cmap):
    plt.figure(figsize=(12, 5))
    colors = cmap(np.linspace(0, 1, trials.shape[0]))
    for i, tr in enumerate(trials):
        n = min(tr.size, time_zeroed.size)
        plt.plot(time_zeroed[:n], tr[:n], color=colors[i], alpha=0.5, lw=0.6)
    plt.axvline(0, color='black', ls='--', label='Picture onset')
    plt.xlabel("Time (ms)")
    plt.ylabel("Pupil size (A.U.)")
    plt.title(title)
    plt.tight_layout()
    plt.show()

if positive_trials.size > 0:
    plot_all_traces(positive_trials, "All Pupil Size Curves — Positive Valence Trials (Picture Onset)", cm.viridis)
if negative_trials.size > 0:
    plot_all_traces(negative_trials, "All Pupil Size Curves — Negative Valence Trials (Picture Onset)", cm.plasma)

# ==============================
# 4) Distribution of pupil values per participant (violin)
# ==============================
# Merge each participant’s trials across both blocks (dominant eye respected above)
participant_pupil_values = []
for subj in range(n_subjects):
    vals = []
    for blk in (0, 1):
        bidx = subj * 2 + blk
        dom  = dominantEye[bidx]
        pup_blk = pupilSizeLeft[bidx] if dom == 0 else pupilSizeRight[bidx]
        pup_blk = np.asarray(pup_blk, float)
        pup_blk[(pup_blk == 0) | (pup_blk == 32700)] = np.nan
        vals.append(pup_blk)
    subj_trials = np.concatenate(vals, axis=0)   
    arr = subj_trials.ravel()
    arr = arr[~np.isnan(arr)]
    participant_pupil_values.append(arr)

# Long df for seaborn
long_df = {
    "Participant": [],
    "Pupil size": []
}
for i, arr in enumerate(participant_pupil_values):
    long_df["Participant"].extend([i+1]*arr.size)  
    long_df["Pupil size"].extend(arr)

violin_df = pd.DataFrame(long_df)

plt.figure(figsize=(14, 5.5))
sns.violinplot(x="Participant", y="Pupil size", data=violin_df, scale='width', inner='quartile', cut=0)
plt.xticks(rotation=90)
plt.title("Pupil Size Distributions per Participant")
plt.tight_layout()
plt.show()

