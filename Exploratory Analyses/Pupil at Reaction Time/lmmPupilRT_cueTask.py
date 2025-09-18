import numpy as np
import pandas as pd
import scipy.io as sio
import os, platform
# -----------------------------------------------------
# checking pymer4 connection with R installation
# -----------------------------------------------------
os.environ['R_HOME'] = r"C:/PROGRA~1/R/R-42~1.1"  
print("R_HOME:", os.environ.get("R_HOME"))
print("Platform:", platform.system())

from pymer4.models import Lmer

# === Load data ===
base_path = r"C:/Users/charl/OneDrive/Dokumente/Uni/7. Semester/Bachelor Thesis/ArrowTask/"

pupilR = sio.loadmat(base_path + "pupilSizeRight_epoched_RTLocked.mat")['pupilSizeRight']
pupilL = sio.loadmat(base_path + "pupilSizeLeft_epoched_RTLocked.mat")['pupilSizeLeft']
first_val = sio.loadmat(base_path + "firstFixationValence.mat")["firstFixationValence"]
last_val = sio.loadmat(base_path + "lastFixationValence.mat")["lastFixationValence"]
blockOrder = sio.loadmat(base_path + "blockOrder_clean.mat")["blockOrder"]
dominantEye = sio.loadmat(base_path + "dominantEye_clean.mat")["dominantEye"]

# === Constants ===
pupilIndex = 500        # index 500 is the reaction time
n_blocks, n_trials, n_time = pupilR.shape

# === Initialize lists ===
participant_list = []
valence_list = []
previous_valence_list = []
condition_list = []
pupilRT_list = []

# === Loop through blocks and trials ===
for b in range(n_blocks):
    # two blocks per participant
    participant = b // 2 + 1
    order = blockOrder[0, b]

    for trial in range(n_trials):
        if trial == 0:
            continue  # skip first trial, no previous

        cur_val = first_val[b, trial]
        prev_val = last_val[b, trial - 1]

        # check valid values
        if np.isnan(cur_val) or np.isnan(prev_val):
            continue

        # select correct eye
        if dominantEye[0, b] == 0:
            pupilData = pupilR
        else:
            pupilData = pupilL

        trial_data = pupilData[b, trial, :].astype(float)
        trial_data[(trial_data == 0) | (trial_data == 32700)] = np.nan

        if pupilIndex >= trial_data.size:
            continue

        pupil_val = trial_data[pupilIndex]
        if np.isnan(pupil_val):
            continue

        # task condition: depends on block order and trial index
        if order == 'A':
            cond = 'congruent' if trial < 44 else 'incongruent'
        else:
            cond = 'incongruent' if trial < 44 else 'congruent'

        # append to lists
        participant_list.append(f"subj_{participant}")
        valence_list.append("pos" if int(cur_val) == 1 else "neg")
        previous_valence_list.append("pos" if int(prev_val) == 1 else "neg")
        condition_list.append(cond)
        pupilRT_list.append(pupil_val)

# === Build DataFrame ===
df = pd.DataFrame({
    "participant": participant_list,
    "valence": valence_list,
    "previous_valence": previous_valence_list,
    "condition": condition_list,
    "pupilRT": pupilRT_list
})

print("Final dataframe shape:", df.shape)
print(df.head())

# ============== calculate Mean, Max and Min Pupil size at RT for each task condition =======================

for cond in sorted(df["condition"].dropna().unique()):
    s = pd.to_numeric(df.loc[df["condition"] == cond, "pupilRT"], errors="coerce").dropna()
    n = len(s)
    mean = np.mean(s) if n else np.nan
    sd   = np.std(s, ddof=1) if n > 1 else np.nan
    vmin = np.min(s) if n else np.nan
    vmax = np.max(s) if n else np.nan
    print(f"{cond:<12} -> n={n}, mean={mean:.3f}, sd={sd:.3f}, min={vmin:.3f}, max={vmax:.3f}")



# ============== LMM =======================

m_with_task = Lmer("pupilRT ~ valence + previous_valence + condition + (1|participant)", data=df)
print("\n=== LMM: pupilRT ~ valence + previous_valence + condition + (1|participant) ===")
print(m_with_task.fit(REML=False))

