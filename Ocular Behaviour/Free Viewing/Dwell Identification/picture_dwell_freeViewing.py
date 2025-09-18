import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# ===== Load =====
xRight = sio.loadmat(r"C:/Users/charl/OneDrive/Dokumente/Uni/7. Semester/Bachelor Thesis/Eye Data Experiment1Task2/Experiment1_Task2/xPositionRight.mat")["xPositionRight"]
xLeft  = sio.loadmat(r"C:/Users/charl/OneDrive/Dokumente/Uni/7. Semester/Bachelor Thesis/Eye Data Experiment1Task2/Experiment1_Task2/xPositionLeft.mat")["xPositionLeft"]
tVec   = sio.loadmat(r"C:/Users/charl/OneDrive/Dokumente/Uni/7. Semester/Bachelor Thesis/Eye Data Experiment1Task2/Experiment1_Task2/timeVector.mat")["timeVector"]
dominantEye = pd.read_csv(r"C:/Users/charl/OneDrive/Dokumente/Uni/7. Semester/Bachelor Thesis/Eye Data Experiment1Task2/Experiment1_Task2/dominantEye.csv", header=None).values.flatten()
valenceRight = pd.read_csv(r"C:/Users/charl/OneDrive/Dokumente/Uni/7. Semester/Bachelor Thesis/Eye Data Experiment1Task2/Experiment1_Task2/valenceRightEye.csv", header=None)
valenceLeft  = pd.read_csv(r"C:/Users/charl/OneDrive/Dokumente/Uni/7. Semester/Bachelor Thesis/Eye Data Experiment1Task2/Experiment1_Task2/valenceLeftEye.csv", header=None)
calib = sio.loadmat(r"C:/Users/charl/OneDrive/Dokumente/Uni/7. Semester/Bachelor Thesis/Eye Data Experiment1Task2/Continuos/calibrationShifts.mat")["calibrationShifts"]

# ===== Trim (participants + first 4 test trials) =====
participants_to_remove = [42, 43, 120, 121]
xRight = np.delete(xRight, participants_to_remove, axis=0)
xLeft  = np.delete(xLeft,  participants_to_remove, axis=0)
tVec   = np.delete(tVec,   participants_to_remove, axis=0)
dominantEye = np.delete(dominantEye, participants_to_remove, axis=0)
calib  = np.delete(calib,  participants_to_remove, axis=0)

xRight = xRight[:, 4:, :]
xLeft  = xLeft[:,  4:, :]
tVec   = tVec[:,   4:, :]
calib  = calib[:,  4:]

P, T, _ = xRight.shape

# ===== Params =====
picture_onset_index = 500
vel_thr, stab_thr, win = 5, 50, 20
central_threshold, margin = 960.0, 100

# ===== Outputs =====
first_start = np.full((P, T), np.nan)
first_end   = np.full((P, T), np.nan)
last_start  = np.full((P, T), np.nan)
last_end    = np.full((P, T), np.nan)
first_side  = np.full((P, T), np.nan)  
last_side   = np.full((P, T), np.nan)  
last_val    = np.full((P, T), np.nan)  

# ===== Detect first dwell per trial =====
for p in range(P):
    for tr in range(T):
        x = xRight[p, tr, :] if dominantEye[p] == 1 else xLeft[p, tr, :]
        t = tVec[p, tr, :].astype(float)
        c = calib[p, tr]
        if not np.isfinite(c): 
            continue
        low, high = c - margin, c + margin
        v = np.abs(np.diff(x))

        saccade_out = False
        start_ms = None
        start_idx = None

        for i in range(picture_onset_index, len(x) - win):
            xv = x[i]
            if not saccade_out:
                if i > 0 and np.isfinite(v[i-1]) and v[i-1] > vel_thr and (xv < low or xv > high):
                    saccade_out = True
                    continue
            else:
                w = x[i:i+win]
                if (np.nanmax(w) - np.nanmin(w)) <= stab_thr:
                    start_ms = t[i]
                    start_idx = i
                    first_side[p, tr] = 0.0 if xv < c else 1.0
                    break

        if start_ms is None:
            continue

        last_stable = None
        end_ms = t[-1]
        direction = "left" if x[start_idx] < central_threshold else "right"
        for j in range(start_idx + 1, len(x) - win):
            wprev = x[j-win:j]
            if (np.nanmax(wprev) - np.nanmin(wprev)) <= stab_thr:
                last_stable = t[j-3]
            back_to_center = ((direction == "left" and x[j] > (central_threshold - margin)) or
                              (direction == "right" and x[j] < (central_threshold + margin)))
            if np.isfinite(v[j-1]) and v[j-1] > vel_thr and back_to_center:
                end_ms = last_stable if last_stable is not None else t[j-1]
                break

        first_start[p, tr] = start_ms
        first_end[p, tr]   = end_ms

# ===== Iteratively find last dwell per trial =====
for p in range(P):
    for tr in range(T):
        if not (np.isfinite(first_start[p, tr]) and np.isfinite(first_end[p, tr])):
            continue

        x = xRight[p, tr, :] if dominantEye[p] == 1 else xLeft[p, tr, :]
        t = tVec[p, tr, :].astype(float)
        c = calib[p, tr]
        low, high = c - margin, c + margin
        v = np.abs(np.diff(x))

        # initialize last = first
        last_start[p, tr] = first_start[p, tr]
        last_end[p, tr]   = first_end[p, tr]
        last_side[p, tr]  = first_side[p, tr]

        # resume search after first_end
        idx_end = np.argmin(np.abs(t - last_end[p, tr]))
        cur = idx_end

        while True:
            sacc, new_start_idx = False, None
            for i in range(cur + 1 + 20, len(x) - win):
                xv = x[i]
                if not sacc:
                    if i > 0 and np.isfinite(v[i-1]) and v[i-1] > vel_thr and (xv < low or xv > high):
                        sacc = True
                        continue
                else:
                    w = x[i:i+win]
                    if (np.nanmax(w) - np.nanmin(w)) <= stab_thr:
                        new_start_idx = i
                        break

            if new_start_idx is None:
                break

            # close this new dwell
            direction = "left" if x[new_start_idx] < central_threshold else "right"
            last_stable, new_end_ms = None, t[-1]
            for j in range(new_start_idx + 1, len(x) - win):
                wprev = x[j-win:j]
                if (np.nanmax(wprev) - np.nanmin(wprev)) <= stab_thr:
                    last_stable = t[j-3]
                back_to_center = ((direction == "left" and x[j] > (central_threshold - margin)) or
                                  (direction == "right" and x[j] < (central_threshold + margin)))
                if np.isfinite(v[j-1]) and v[j-1] > vel_thr and back_to_center:
                    new_end_ms = last_stable if last_stable is not None else t[j-1]
                    break

            last_start[p, tr] = t[new_start_idx]
            last_end[p, tr]   = new_end_ms
            last_side[p, tr]  = 0.0 if x[new_start_idx] < c else 1.0
            cur = np.argmin(np.abs(t - new_end_ms))

print("✅ Picture dwells detected (first & last).")

# ===== Last-dwell valence ====
# dominantEye is per *block*, so participants are grouped in pairs of blocks (2*59 = 118; 88 trials each)
for participant in range(59):
    dom_eye = dominantEye[participant * 2]  # block 1 of that participant
    for trial in range(176):  # 2 blocks × 88 trials
        val = valenceLeft.iloc[participant, trial] if dom_eye == 0 else valenceRight.iloc[participant, trial]
        if trial < 88:
            b = participant * 2
            tr = trial
        else:
            b = participant * 2 + 1
            tr = trial - 88

        a = first_side[b, tr]
        d = last_side[b, tr]
        if not (np.isfinite(a) and np.isfinite(d)):
            continue

        # if same side: last valence equals that side's valence; else flip (1 -> 0, 0 -> 1)
        if val in (0, 1):
            last_val[b, tr] = val if a == d else (0 if val == 1 else 1)

print("✅ Last-dwell valence computed.")

# ===== Save =====
sio.savemat(r"C:/Users/charl/OneDrive/Dokumente/Uni/7. Semester/Bachelor Thesis/Eye Data Experiment1Task2/fixationPictureStartTimes.mat",
            {"fixationStartTimes": first_start})
sio.savemat(r"C:/Users/charl/OneDrive/Dokumente/Uni/7. Semester/Bachelor Thesis/Eye Data Experiment1Task2/fixationPictureEndTimes.mat",
            {"fixationEndTimes": first_end})
sio.savemat(r"C:/Users/charl/OneDrive/Dokumente/Uni/7. Semester/Bachelor Thesis/Eye Data Experiment1Task2/firstFixationSide.mat",
            {"firstFixationSide": first_side})
sio.savemat(r"C:/Users/charl/OneDrive/Dokumente/Uni/7. Semester/Bachelor Thesis/Eye Data Experiment1Task2/lastFixationSide.mat",
            {"lastFixationSide": last_side})
sio.savemat(r"C:/Users/charl/OneDrive/Dokumente/Uni/7. Semester/Bachelor Thesis/Eye Data Experiment1Task2/lastFixationPictureStartTimes.mat",
            {"lastFixationPictureStartTimes": last_start})
sio.savemat(r"C:/Users/charl/OneDrive/Dokumente/Uni/7. Semester/Bachelor Thesis/Eye Data Experiment1Task2/lastFixationPictureEndTimes.mat",
            {"lastFixationPictureEndTimes": last_end})
sio.savemat(r"C:/Users/charl/OneDrive/Dokumente/Uni/7. Semester/Bachelor Thesis/Eye Data Experiment1Task2/lastFixationValence.mat",
            {"lastFixationValence": last_val})

print("💾 Saved first/last dwell times, sides, and last valence.")


##############################################################################################################################################################################################################
# ===== Plot  =====
##############################################################################################################################################################################################################

def plot_trial(participant_number=9, trial_number=26):
    p = participant_number - 1
    tr = trial_number - 1
    x = xRight[p, tr, :] if dominantEye[p] == 1 else xLeft[p, tr, :]
    t = tVec[p, tr, :].astype(float)
    c = calib[p, tr]
    low, high = c - margin, c + margin

    fs1, fe1 = first_start[p, tr], first_end[p, tr]
    fsl, fel = last_start[p, tr], last_end[p, tr]

    t_rel = t - t[picture_onset_index]

    plt.figure(figsize=(10, 5))
    plt.plot(t_rel, x, color='black', lw=1.0, label="X Position")
    plt.axvline(0, color='red', ls='--', label="Picture Onset")
    plt.axhline(low,  color='tab:blue', ls='-.', label="Lower Threshold")
    plt.axhline(high, color='tab:blue', ls='-.', label="Upper Threshold")

    if np.isfinite(fs1) and np.isfinite(fe1) and fe1 >= fs1:
        m1 = (t >= fs1) & (t <= fe1)
        if np.any(m1): plt.plot(t_rel[m1], x[m1], color='tab:red', lw=2.0, label="First Dwell on Picture")
    if np.isfinite(fsl) and np.isfinite(fel) and fel >= fsl:
        m2 = (t >= fsl) & (t <= fel)
        if np.any(m2): plt.plot(t_rel[m2], x[m2], color='orange', lw=2.0, label="Last Dwell on Picture")

    plt.xlabel("Time (ms) relative to picture onset")
    plt.ylabel("X Position")
    plt.title(f"X Position (Participant {participant_number}, Trial {trial_number})")
    plt.legend(loc="best")
    plt.tight_layout()
    plt.show()



