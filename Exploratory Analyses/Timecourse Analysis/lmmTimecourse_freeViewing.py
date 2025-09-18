import numpy as np
import pandas as pd
import scipy.io as sio
import os 
import platform
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats


os.environ['R_HOME'] = r"C:/PROGRA~1/R/R-42~1.1"

from pymer4.models import Lmer  # Required for Lmer models

print("R_HOME:", os.environ.get("R_HOME"))
print("Platform:", platform.system())

# === Load all relevant data ===
dominantEye = pd.read_csv(
    "C:/Users/charl/OneDrive/Dokumente/Uni/7. Semester/Bachelor Thesis/"
    "Eye Data Experiment1Task2/Experiment1_Task2/dominantEye.csv", header=None
).values.flatten()

firstvalenceLeft = pd.read_csv(
    "C:/Users/charl/OneDrive/Dokumente/Uni/7. Semester/Bachelor Thesis/"
    "Eye Data Experiment1Task2/Experiment1_Task2/valenceLeftEye.csv", header=None
).values

firstvalenceRight = pd.read_csv(
    "C:/Users/charl/OneDrive/Dokumente/Uni/7. Semester/Bachelor Thesis/"
    "Eye Data Experiment1Task2/Experiment1_Task2/valenceRightEye.csv", header=None
).values

# === Load Last Fixation Valence Matrix ===
last_valence_mat = sio.loadmat("C:/Users/charl/OneDrive/Dokumente/Uni/7. Semester/Bachelor Thesis/Eye Data Experiment1Task2/lastFixationValence.mat")["lastFixationValence"]

pupilSizeRight = sio.loadmat(
     "C:/Users/charl/OneDrive/Dokumente/Uni/7. Semester/Bachelor Thesis/Eye Data Experiment1Task2/Experiment1_Task2/PicturePupilRight_epoched.mat"
)['PupilSizeRight']

pupilSizeLeft = sio.loadmat(
    "C:/Users/charl/OneDrive/Dokumente/Uni/7. Semester/Bachelor Thesis/Eye Data Experiment1Task2/Experiment1_Task2/PicturePupilLeft_epoched.mat"
)['PupilSizeLeft']

timeVector = sio.loadmat(
    "C:/Users/charl/OneDrive/Dokumente/Uni/7. Semester/Bachelor Thesis/Eye Data Experiment1Task2/Experiment1_Task2/PictureTimeVector_epoched.mat"
)['timeVector']

fullTrialPupil = sio.loadmat("C:/Users/charl/OneDrive/Dokumente/Uni/7. Semester/Bachelor Thesis/Eye Data Experiment1Task2/Continuos/epochedPupilSize.mat")['epochedPupilSize']
fullTrialTimeVector = sio.loadmat("C:/Users/charl/OneDrive/Dokumente/Uni/7. Semester/Bachelor Thesis/Eye Data Experiment1Task2/Continuos/epochedTimeVector.mat")['epochedTimeVector']

picID_positive = pd.read_csv("C:/Users/charl/OneDrive/Dokumente/Uni/7. Semester/Bachelor Thesis/Eye Data Experiment1Task2/positiveSequence.csv", header=None)
picID_negative = pd.read_csv("C:/Users/charl/OneDrive/Dokumente/Uni/7. Semester/Bachelor Thesis/Eye Data Experiment1Task2/negativeSequence.csv", header=None)
picOnset = pd.read_csv("C:/Users/charl/OneDrive/Dokumente/Uni/7. Semester/Bachelor Thesis/Eye Data Experiment1Task2/Continuos/pictureOnsetTimes.csv")

arousal = pd.read_excel("C:/Users/charl/OneDrive/Dokumente/Uni/7. Semester/Bachelor Thesis/Eye Data Experiment1Task2/Experiment1_Task2/median_arousal_ratings.xlsx")

# === Remove excluded participants ===
participants_to_remove = [42, 43, 120, 121]
dominantEye = np.delete(dominantEye, participants_to_remove, axis=0)
picID_negative = np.delete(picID_negative, participants_to_remove, axis=1)
picID_positive = np.delete(picID_positive, participants_to_remove, axis=1)

trials_to_remove = [0,1,2,3]
picID_negative = np.delete(picID_negative, trials_to_remove, axis=0)
picID_positive = np.delete(picID_positive, trials_to_remove, axis=0)
fullTrialPupil = np.delete(fullTrialPupil, trials_to_remove, axis=1)
fullTrialTimeVector = np.delete(fullTrialTimeVector, trials_to_remove, axis=1)
picOnset = np.delete(picOnset, trials_to_remove, axis=1)

# === Remove test trials (first 4)
pupilSizeLeft = pupilSizeLeft[:, 4:, :]
pupilSizeRight = pupilSizeRight[:, 4:, :]
timeVector = timeVector[:, 4:, :]

# === Constants ===
n_participants = 59 
n_trials = 176
# Define indexes of timepoints of interest (e.g., every 50 indices ~100 ms steps)
timepoints_indices = [i * 50 for i in range(25)]  

# === Lists to store only the necessary info ===
participantID = []
current_valence = []
prev_valences = []
pupil_timepoints = [[] for _ in range(len(timepoints_indices))]

for p in range(n_participants):
    block1 = p * 2
    block2 = p * 2 + 1
    for trial_idx in range(n_trials):
        # Skip first trial of each block (no previous)
        if trial_idx == 0 or trial_idx == 88:
            continue
        
        if dominantEye[block1] == 0:  # dominant eye = LEFT
            valence = firstvalenceLeft[p, trial_idx]
            if trial_idx < 88:  # block 1
                pupilAll = pupilSizeLeft[block1, trial_idx]
                prev_val = last_valence_mat[block1, trial_idx - 1]
            else:  # block 2
                pupilAll = pupilSizeLeft[block2, trial_idx - 88]
                prev_val = last_valence_mat[block2, trial_idx - 89]
        else:  # dominant eye = RIGHT
            valence = firstvalenceRight[p, trial_idx]
            if trial_idx < 88:  # block 1
                pupilAll = pupilSizeRight[block1, trial_idx]
                prev_val = last_valence_mat[block1, trial_idx - 1]
            else:  # block 2
                pupilAll = pupilSizeRight[block2, trial_idx - 88]
                prev_val = last_valence_mat[block2, trial_idx - 89]

        # Extract pupil size at each selected timepoint
        for idx_tp, tp_idx in enumerate(timepoints_indices):
            if tp_idx < len(pupilAll):
                pupil_timepoints[idx_tp].append(pupilAll[tp_idx])
            else:
                pupil_timepoints[idx_tp].append(np.nan)

        # Store trial-level metadata
        participantID.append(f"subj_{p}")
        current_valence.append(str(valence))
        prev_valences.append(str(prev_val))


# Build DataFrame including pupil size columns for each timepoint
df_lmm_REAL = pd.DataFrame({
    "participant": participantID,
    "valence": current_valence,
    "previous_valence": prev_valences
})

# Add pupil size columns dynamically
for i in range(len(timepoints_indices)):
    df_lmm_REAL[f"pupilSize_{i}"] = pupil_timepoints[i]

print(df_lmm_REAL.head())

# Convert to appropriate types for pymer4
df_lmm_REAL['participant'] = df_lmm_REAL['participant'].astype('category')
df_lmm_REAL['valence'] = df_lmm_REAL['valence'].astype('category')
df_lmm_REAL['previous_valence'] = df_lmm_REAL['previous_valence'].astype('category')

# Function to fit LMMs across all timepoints and collect p-values
def run_models_and_collect_pvals(df, formula_template, timepoints_count, group_label):
    pvals_valence = [np.nan] * timepoints_count
    pvals_previous_valence = [np.nan] * timepoints_count
    results = []
    
    for i in range(timepoints_count):
        col_name = f"pupilSize_{i}"
        df_timepoint = df.dropna(subset=[col_name])
        if df_timepoint.shape[0] == 0:
            print(f"No data for {group_label} at {col_name}, skipping.")
            continue

        formula = formula_template.format(i=i)
        model = Lmer(formula, data=df_timepoint)
        result = model.fit(REML=False)
        print(f"\n=== {group_label} Model for {col_name}: {formula} ===")
        print(result)
        results.append(result)


        # Extract p-values safely with fallback keys
        valence_key = next((k for k in result.index if 'valence' in k), None)
        prev_val_key = next((k for k in result.index if 'previous_valence' in k), None)

        if valence_key:
            pvals_valence[i] = result.loc[valence_key, 'P-val']
        else:
            print(f"No valence coefficient found in model {i}")

        if prev_val_key:
            pvals_previous_valence[i] = result.loc[prev_val_key, 'P-val']
        else:
            print(f"No previous_valence coefficient found in model {i}")


    return pvals_valence, pvals_previous_valence, results

# Define formula template
formula_template = "pupilSize_{i} ~ valence * previous_valence + (1|participant)"

# Run models and get p-values
pvals_valence, pvals_previous_valence, model_results = run_models_and_collect_pvals(df_lmm_REAL, formula_template, len(timepoints_indices), "All Participants")

# Prepare time axis for plotting (assuming 100 ms steps per 50 indices)
timepoints_ms = [i * 100 for i in range(len(timepoints_indices))]
shifted_timepoints = [t - 1000 for t in timepoints_ms]  # center picture onset at 0 ms

# Plot p-values over time
plt.figure(figsize=(10,6))
plt.plot(shifted_timepoints, pvals_valence, marker='o', linestyle='-', label='Valence')
plt.plot(shifted_timepoints, pvals_previous_valence, marker='s', linestyle='--', label='Previous Valence')

plt.axhline(0.05, color='red', linestyle='--', label='p = 0.05 threshold')
plt.axvline(0, color='black', linestyle='--', linewidth=1, label='Picture Onset')

plt.xlabel('Time (ms) relative to Picture Onset')
plt.ylabel('p-value')
plt.title('P-values of fixed effects over time (All Participants)')
plt.legend()
plt.show()
