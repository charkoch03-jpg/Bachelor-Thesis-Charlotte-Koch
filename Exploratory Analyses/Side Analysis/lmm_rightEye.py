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
first_val = sio.loadmat(base_path + "firstFixationValence.mat")["firstFixationValence"]
last_val = sio.loadmat(base_path + "lastFixationValence.mat")["lastFixationValence"]
blockOrder = sio.loadmat(base_path + "blockOrder_clean.mat")["blockOrder"]
first_side = sio.loadmat(base_path + "firstFixationSide.mat")["firstFixationSide"]

# === Constants ===
pictureOnsetIdx = 500   # 0 ms
pupilIndex = 750        # +500 ms post-onset (500 Hz sampling)
n_blocks, n_trials, n_time = pupilR.shape

# === Initialize lists ===
participant_list = []
valence_list = []
previous_valence_list = []
condition_list = []
side_list = []
pupil500_list = []

# === Loop through blocks and trials ===
for b in range(n_blocks):
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

        trial_data = pupilR[b, trial, :].astype(float)

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
        side_list.append("left" if int(first_side[b, trial]) == 0 else "right")
        pupil500_list.append(pupil_val)

# === Build DataFrame ===
df = pd.DataFrame({
    "participant": participant_list,
    "valence": valence_list,
    "previous_valence": previous_valence_list,
    "condition": condition_list,
    "side": side_list,
    "pupil500": pupil500_list
})

# Convert to categorical
for col in ["participant", "valence", "previous_valence", "condition", "side"]:
    df[col] = df[col].astype('category')

print("Final dataframe shape:", df.shape)
print(df.head())


# ===================== Descriptive Stats by SIDE =====================
print("\n=== Stats by SIDE ===")
for s in df["side"].unique():
    sub = df[df["side"] == s]["pupil500"].dropna()
    print(s,
          "n=", len(sub),
          "mean=", np.round(sub.mean(), 3),
          "sd=", np.round(sub.std(), 3),
          "min=", np.round(sub.min(), 3),
          "max=", np.round(sub.max(), 3))

# ===================== LMM =====================

m_with_side = Lmer("pupil500 ~ valence + previous_valence + condition + side + (1|participant)", data=df)
print("\n=== LMM: pupil500 ~ valence + previous_valence + condition + side + (1|participant) ===")
print(m_with_side.fit(REML=False))
