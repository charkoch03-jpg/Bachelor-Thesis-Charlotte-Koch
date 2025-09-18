import numpy as np
import pandas as pd
import scipy.io as sio
import os, platform
import matplotlib.pyplot as plt
from pymer4.models import Lmer

# -----------------------------------------------------
# R setup
# -----------------------------------------------------
os.environ['R_HOME'] = r"C:/PROGRA~1/R/R-42~1.1"
print("R_HOME:", os.environ.get("R_HOME"))
print("Platform:", platform.system())

# === Load data ===
base_path = r"C:/Users/charl/OneDrive/Dokumente/Uni/7. Semester/Bachelor Thesis/ArrowTask/"

pupilR = sio.loadmat(base_path + "pupilSizeRightPic_clean.mat")["pupilSizeRight"]
pupilL = sio.loadmat(base_path + "pupilSizeLeftPic_clean.mat")["pupilSizeLeft"]
first_val = sio.loadmat(base_path + "firstFixationValence.mat")["firstFixationValence"]
last_val = sio.loadmat(base_path + "lastFixationValence.mat")["lastFixationValence"]
blockOrder = sio.loadmat(base_path + "blockOrder_clean.mat")["blockOrder"]
dominantEye = sio.loadmat(base_path + "dominantEye_clean.mat")["dominantEye"]
cheater = sio.loadmat(base_path + "cheater.mat")["cheater"]  

# === Constants ===
n_blocks, n_trials, n_time = pupilR.shape
timepoints_indices = list(range(0, 1501, 50)) 

# === Initialize lists ===
participant_list = []
valence_list = []
previous_valence_list = []
condition_list = []
pupil_timepoints = {tp: [] for tp in timepoints_indices}

# === Loop through blocks and trials ===
for b in range(n_blocks):
    participant = b // 2 + 1
    order = blockOrder[0, b]

    for trial in range(n_trials):
        if trial == 0:
            continue  # skip first trial, no previous
        if cheater[b, trial] != 1:
            continue  # only include cheater trials

        cur_val = first_val[b, trial]
        prev_val = last_val[b, trial - 1]

        if np.isnan(cur_val) or np.isnan(prev_val):
            continue

        pupilData = pupilR if dominantEye[0, b] == 0 else pupilL
        trial_data = pupilData[b, trial, :].astype(float)

        if max(timepoints_indices) >= trial_data.size:
            continue

        # extract pupil size at all timepoints
        for tp in timepoints_indices:
            pupil_timepoints[tp].append(trial_data[tp])

        # task condition based on block order and trial index
        if order == 'A':
            cond = 'congruent' if trial < 44 else 'incongruent'
        else:
            cond = 'incongruent' if trial < 44 else 'congruent'

        # append participant and fixed effects info
        participant_list.append(f"subj_{participant}")
        valence_list.append("pos" if int(cur_val) == 1 else "neg")
        previous_valence_list.append("pos" if int(prev_val) == 1 else "neg")
        condition_list.append(cond)

# === Build DataFrame ===
df_lmm = pd.DataFrame({
    "participant": participant_list,
    "valence": valence_list,
    "previous_valence": previous_valence_list,
    "condition": condition_list
})

for tp in timepoints_indices:
    df_lmm[f"pupil_{tp}"] = pupil_timepoints[tp]

# convert to categorical
for col in ["participant", "valence", "previous_valence", "condition"]:
    df_lmm[col] = df_lmm[col].astype('category')

print("Final dataframe shape:", df_lmm.shape)
print(df_lmm.head())

# === Run LMMs for each timepoint and collect p-values ===
fixed_effects = ['valence', 'previous_valence', 'condition']
pvals = {fe: [] for fe in fixed_effects}

for tp in timepoints_indices:
    col_name = f"pupil_{tp}"
    df_time = df_lmm.dropna(subset=[col_name])
    
    if df_time.shape[0] == 0:
        for fe in fixed_effects:
            pvals[fe].append(np.nan)
        continue

    formula = f"{col_name} ~ valence + previous_valence + condition + (1|participant)"
    model = Lmer(formula, data=df_time)
    res = model.fit(REML=False)

    for fe in fixed_effects:
        key = next((k for k in res.index if fe in k), None)
        pvals[fe].append(res.loc[key, 'P-val'] if key else np.nan)

# === Plot p-values over time ===
plt.figure(figsize=(10, 6))
for fe in fixed_effects:
    plt.plot(timepoints_indices, pvals[fe], marker='o', label=fe)

plt.axhline(0.05, color='red', linestyle='--', label='p = 0.05 threshold')
plt.axvline(0, color='black', linestyle='--', linewidth=1, label='Picture Onset')
plt.xlabel('Time (ms) relative to Picture Onset')
plt.ylabel('p-value')
plt.title('P-values of fixed effects over time')
plt.legend()
plt.show()
