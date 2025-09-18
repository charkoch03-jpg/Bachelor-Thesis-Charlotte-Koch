import os
import platform
import numpy as np
import pandas as pd
import scipy.io as sio
from pymer4.models import Lmer

# -----------------------------------------------------
# checking pymer4 connection with R installation
# -----------------------------------------------------
os.environ['R_HOME'] = r"C:/PROGRA~1/R/R-42~1.1"  
print("R_HOME:", os.environ.get("R_HOME"))
print("Platform:", platform.system())

# ==============================
# Paths & data load
# ==============================
base_path = r"C:/Users/charl/OneDrive/Dokumente/Uni/7. Semester/Bachelor Thesis/Eye Data Experiment1Task2/Experiment1_Task2/"

# CSVs
dominantEye = pd.read_csv(base_path + "dominantEye.csv", header=None).values.flatten()   
valenceLeft  = pd.read_csv(base_path + "valenceLeftEye.csv",  header=None).values        
valenceRight = pd.read_csv(base_path + "valenceRightEye.csv", header=None).values        

# MATs
pupilR = sio.loadmat(base_path + "PicturePupilRight_epoched.mat")["PupilSizeRight"]      
pupilL = sio.loadmat(base_path + "PicturePupilLeft_epoched.mat")["PupilSizeLeft"]        
timeV  = sio.loadmat(base_path + "PictureTimeVector_epoched.mat")["timeVector"]  
last_val = sio.loadmat(base_path + "lastFixationValence.mat")["lastFixationValence"] 

# ==============================
# Preparation
# ==============================
# remove participants
participants_to_remove = [42, 43, 120, 121]
dominantEye = np.delete(dominantEye, participants_to_remove, axis=0)

# Drop test trials
pupilL = pupilL[:, 4:, :]
pupilR = pupilR[:, 4:, :]
timeV  = timeV[:, 4:, :]


# === Constants ===
n_participants = 59 
n_trials = 176    
picture_onset_idx = 500                                               

# --- lists for the dataframe ---
valence_list = []
prev_valence_list = []
pupil500_list = []   
participantID = []

pupil_idx = 750 

for p in range(n_participants):
    block1 = p * 2
    block2 = p * 2 + 1

    for trial_idx in range(n_trials):
        # Skip first trial of each block (no previous)
        if trial_idx == 0 or trial_idx == 88:
            continue

        # pick dominant eye 
        if dominantEye[block1] == 0:
            # values from LEFT eye source
            cur_val = valenceLeft[p, trial_idx]
            if trial_idx < 88:
                # block 1
                y = pupilL[block1, trial_idx, pupil_idx]
                prev_val = last_val[block1, trial_idx - 1]
            else:
                # block 2
                y = pupilL[block2, trial_idx - 88, pupil_idx]
                prev_val = last_val[block2, trial_idx - 89]
        else:
            # values from RIGHT eye sources
            cur_val = valenceRight[p, trial_idx]
            if trial_idx < 88:
                # block 1
                y = pupilR[block1, trial_idx, pupil_idx]
                prev_val = last_val[block1, trial_idx - 1]
            else:
                # block 2
                y = pupilR[block2, trial_idx - 88, pupil_idx]
                prev_val = last_val[block2, trial_idx - 89]

        # basic validity check (skip nan)
        if np.isnan(cur_val) or np.isnan(prev_val) or np.isnan(y):
            continue

        # append to lists 
        valence_list.append(int(cur_val))         
        prev_valence_list.append(int(prev_val))    
        pupil500_list.append(float(y))
        participantID.append(f"subj_{p}")

# ==============================
# Make the DataFrame
# ==============================
df_lmm = pd.DataFrame({
    "participant": participantID,
    "valence": valence_list,
    "previous_valence": prev_valence_list,
    "pupil500": pupil500_list
})

# Optional: convert to categorical 
df_lmm["participant"] = df_lmm["participant"].astype("category")
df_lmm["valence"] = df_lmm["valence"].astype("category")
df_lmm["previous_valence"] = df_lmm["previous_valence"].astype("category")

print("LMM table shape:", df_lmm.shape)
print(df_lmm.head())

# ==============================
# Calculate Descriptive Stats of Pupil Size 
# ==============================

print("\n=== Stats by VALENCE ===")
for v in sorted(df_lmm["valence"].dropna().unique()):
    sub = pd.to_numeric(df_lmm[df_lmm["valence"] == v]["pupil500"], errors="coerce").dropna()
    n = len(sub)
    mean = sub.mean() if n else np.nan
    sd = sub.std(ddof=1) if n > 1 else np.nan
    vmin = sub.min() if n else np.nan
    vmax = sub.max() if n else np.nan
    print(f"valence={v} -> n={n}, mean={mean:.3f}, sd={sd if not np.isnan(sd) else 'nan'}, min={vmin if not np.isnan(vmin) else 'nan'}, max={vmax if not np.isnan(vmax) else 'nan'}")

print("\n=== Stats by PREVIOUS_VALENCE ===")
for pv in sorted(df_lmm["previous_valence"].dropna().unique()):
    sub = pd.to_numeric(df_lmm[df_lmm["previous_valence"] == pv]["pupil500"], errors="coerce").dropna()
    n = len(sub)
    mean = sub.mean() if n else np.nan
    sd = sub.std(ddof=1) if n > 1 else np.nan
    vmin = sub.min() if n else np.nan
    vmax = sub.max() if n else np.nan
    print(f"prev_valence={pv} -> n={n}, mean={mean:.3f}, sd={sd if not np.isnan(sd) else 'nan'}, min={vmin if not np.isnan(vmin) else 'nan'}, max={vmax if not np.isnan(vmax) else 'nan'}")

print("\n=== Stats for VALENCE × PREVIOUS_VALENCE ===")
vals = sorted(df_lmm["valence"].dropna().unique())
prevs = sorted(df_lmm["previous_valence"].dropna().unique())
for v in vals:
    for pv in prevs:
        sub = pd.to_numeric(df_lmm[(df_lmm["valence"] == v) & (df_lmm["previous_valence"] == pv)]["pupil500"], errors="coerce").dropna()
        n = len(sub)
        mean = sub.mean() if n else np.nan
        sd = sub.std(ddof=1) if n > 1 else np.nan
        vmin = sub.min() if n else np.nan
        vmax = sub.max() if n else np.nan
        print(f"valence={v} / prev={pv} -> n={n}, mean={mean:.3f}, sd={sd if not np.isnan(sd) else 'nan'}, min={vmin if not np.isnan(vmin) else 'nan'}, max={vmax if not np.isnan(vmax) else 'nan'}")



# ==============================
# Fit the LMM at +500 ms (index 750)
# ==============================

formula = "pupil500 ~ valence * previous_valence + (1|participant)"
model = Lmer(formula, data=df_lmm)
result = model.fit(REML=False)
print("\n=== LMM @ +500 ms (index 750) ===")
print(result)
