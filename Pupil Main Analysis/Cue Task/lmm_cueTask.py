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

pupilR = sio.loadmat(base_path + "pupilSizeRightPic_clean.mat")["pupilSizeRight"]
pupilL = sio.loadmat(base_path + "pupilSizeLeftPic_clean.mat")["pupilSizeLeft"]
first_val = sio.loadmat(base_path + "firstFixationValence.mat")["firstFixationValence"]
last_val = sio.loadmat(base_path + "lastFixationValence.mat")["lastFixationValence"]
blockOrder = sio.loadmat(base_path + "blockOrder_clean.mat")["blockOrder"]
dominantEye = sio.loadmat(base_path + "dominantEye_clean.mat")["dominantEye"]

# === Constants ===
pictureOnsetIdx = 500   # 0 ms
pupilIndex = 750        # +500 ms post-onset (500 Hz sampling)
n_blocks, n_trials, n_time = pupilR.shape

# === Initialize lists ===
participant_list = []
valence_list = []
previous_valence_list = []
condition_list = []
pupil500_list = []

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
        pupil500_list.append(pupil_val)

# === Build DataFrame ===
df = pd.DataFrame({
    "participant": participant_list,
    "valence": valence_list,
    "previous_valence": previous_valence_list,
    "condition": condition_list,
    "pupil500": pupil500_list
})

print("Final dataframe shape:", df.shape)
print(df.head())

# ============== Calculate Descriptive Stats of Pupil Size =======================

print("\n=== Stats by CONDITION ===")
for cond in df["condition"].unique():
    sub = df[df["condition"] == cond]["pupil500"].dropna()
    print(cond, 
          "n=", len(sub), 
          "mean=", sub.mean(), 
          "sd=", sub.std(), 
          "min=", sub.min(), 
          "max=", sub.max())

print("\n=== Stats by VALENCE ===")
for v in df["valence"].unique():
    sub = df[df["valence"] == v]["pupil500"].dropna()
    print(v, 
          "n=", len(sub), 
          "mean=", sub.mean(), 
          "sd=", sub.std(), 
          "min=", sub.min(), 
          "max=", sub.max())

print("\n=== Stats by PREVIOUS_VALENCE ===")
for pv in df["previous_valence"].unique():
    sub = df[df["previous_valence"] == pv]["pupil500"].dropna()
    print(pv, 
          "n=", len(sub), 
          "mean=", sub.mean(), 
          "sd=", sub.std(), 
          "min=", sub.min(), 
          "max=", sub.max())

print("\n=== Stats for VALENCE × CONDITION ===")
for cond in df["condition"].unique():
    for v in df["valence"].unique():
        sub = df[(df["condition"] == cond) & (df["valence"] == v)]["pupil500"].dropna()
        print("cond=", cond, "valence=", v, 
              "n=", len(sub), 
              "mean=", sub.mean(), 
              "sd=", sub.std(), 
              "min=", sub.min(), 
              "max=", sub.max())

print("\n=== Stats for VALENCE × PREVIOUS_VALENCE ===")
for v in df["valence"].unique():
    for pv in df["previous_valence"].unique():
        sub = df[(df["valence"] == v) & (df["previous_valence"] == pv)]["pupil500"].dropna()
        print("valence=", v, "prev=", pv, 
              "n=", len(sub), 
              "mean=", sub.mean(), 
              "sd=", sub.std(), 
              "min=", sub.min(), 
              "max=", sub.max())


# ============== LMMs =======================

# 1) Valence × Previous Valence (random intercept for participant)
m_vxprev = Lmer("pupil500 ~ valence * previous_valence + (1|participant)", data=df)
print("\n=== LMM: pupil500 ~ valence * previous_valence + (1|participant) ===")
print(m_vxprev.fit(REML=False))

# 2) Add Task condition (main effect)
m_with_task = Lmer("pupil500 ~ valence + previous_valence + condition + (1|participant)", data=df)
print("\n=== LMM: pupil500 ~ valence + previous_valence + condition + (1|participant) ===")
print(m_with_task.fit(REML=False))

