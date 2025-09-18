

import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# ===== Load =====
xRight = sio.loadmat("C:/Users/charl/OneDrive/Dokumente/Uni/7. Semester/Bachelor Thesis/ArrowTask/xPositionRightPic_clean.mat")["xPositionRight"]
xLeft  = sio.loadmat("C:/Users/charl/OneDrive/Dokumente/Uni/7. Semester/Bachelor Thesis/ArrowTask/xPositionLeftPic_clean.mat")["xPositionLeft"]
tVec   = sio.loadmat("C:/Users/charl/OneDrive/Dokumente/Uni/7. Semester/Bachelor Thesis/ArrowTask/timeVectorPic_clean.mat")["timeVector"]
dominantEye = pd.read_csv("C:/Users/charl/OneDrive/Dokumente/Uni/7. Semester/Bachelor Thesis/ArrowTask/dominantEye_clean.mat")['dominantEye']
calib = sio.loadmat("C:/Users/charl/OneDrive/Dokumente/Uni/7. Semester/Bachelor Thesis/ArrowTask/calibrationShifts.mat")["calibrationShifts"]
reactionTimes = sio.loadmat("C:/Users/charl/OneDrive/Dokumente/Uni/7. Semester/Bachelor Thesis/ArrowTask/reactionTimes_clean.mat")["reactionTimes"] 
pictureSequence = sio.loadmat("C:/Users/charl/OneDrive/Dokumente/Uni/7. Semester/Bachelor Thesis/ArrowTask//pictureSequence_clean.mat")["pictureSequence"]


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
first_val   = np.full((P, T), np.nan)  
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

        # valence of first picture dwell
        pic = pictureSequence[p, 0, tr] if first_side[p, tr] == 1.0 else pic == pictureSequence[p, 1, tr]

        first_val[p, tr] == 1 if pic <= 44 else first_val[p, tr] == 0

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

            # valence of last picture dwell 
            pic = pictureSequence[p, 0, tr] if last_side[p, tr] == 1.0 else pic == pictureSequence[p, 1, tr]

            last_val[p, tr] == 1 if pic <= 44 else last_val[p, tr] == 0


print("✅ Picture dwells detected (first & last).")


# ===== Save =====
sio.savemat("C:/Users/charl/OneDrive/Dokumente/Uni/7. Semester/Bachelor Thesis/ArrowTask/fixationPictureStartTimes.mat",
            {"fixationStartTimes": first_start})
sio.savemat("C:/Users/charl/OneDrive/Dokumente/Uni/7. Semester/Bachelor Thesis/ArrowTask/fixationPictureEndTimes.mat",
            {"fixationEndTimes": first_end})
sio.savemat("C:/Users/charl/OneDrive/Dokumente/Uni/7. Semester/Bachelor Thesis/ArrowTask/firstFixationSide.mat",
            {"firstFixationSide": first_side})
sio.savemat("C:/Users/charl/OneDrive/Dokumente/Uni/7. Semester/Bachelor Thesis/ArrowTask/lastFixationSide.mat",
            {"lastFixationSide": last_side})
sio.savemat("C:/Users/charl/OneDrive/Dokumente/Uni/7. Semester/Bachelor Thesis/ArrowTask/lastFixationPictureStartTimes.mat",
            {"lastFixationPictureStartTimes": last_start})
sio.savemat("C:/Users/charl/OneDrive/Dokumente/Uni/7. Semester/Bachelor Thesis/ArrowTask/lastFixationPictureEndTimes.mat",
            {"lastFixationPictureEndTimes": last_end})
sio.savemat("C:/Users/charl/OneDrive/Dokumente/Uni/7. Semester/Bachelor Thesis/ArrowTask/firstFixationValence.mat",
            {"firstFixationValence": first_val})
sio.savemat("C:/Users/charl/OneDrive/Dokumente/Uni/7. Semester/Bachelor Thesis/ArrowTask/lastFixationValence.mat",
            {"lastFixationValence": last_val})

print("💾 Saved first/last dwell times, sides, and last valence.")




# =========================
# PLOT (pick any trial)
# =========================
def plot_trial(participant_number=38, trial_number=43):
    p = participant_number - 1
    t = trial_number - 1
    use_right = (dominantEye[0, p] if dominantEye.ndim == 2 else dominantEye[p]) == 0
    x = np.asarray(xRight[p, t] if use_right else xLeft[p, t]).ravel()
    tt = np.asarray(tVec[p, t], dtype=float).ravel()
    t0 = tt[picture_onset_index]; tt_rel = tt - t0
    c = calib[p, tr]
    lower, upper = c - margin, c + margin

    fs = first_start[p, t]; fe = first_end[p, t]
    ls = last_start[p, t];  le = last_end[p, t]
    lv = last_val[p, t]

    plt.figure(figsize=(10, 5))
    plt.plot(tt_rel, x, color='black', lw=1.0, label="X Position")
    plt.axvline(0, color='red', ls='--', label="Picture Onset")
    plt.axhline(lower, color='tab:blue', ls='-.', label="Lower Threshold")
    plt.axhline(upper, color='tab:blue', ls='-.', label="Upper Threshold")

    if np.isfinite(fs) and np.isfinite(fe) and fe >= fs:
        m = (tt >= fs) & (tt <= fe)
        if np.any(m): plt.plot(tt_rel[m], x[m], color='tab:red', lw=2.0, label="First Dwell on Picture")
    if np.isfinite(ls) and np.isfinite(le) and le >= ls:
        m2 = (tt >= ls) & (tt <= le)
        if np.any(m2):
            lab = f"Last Dwell (Val={int(lv)})" if np.isfinite(lv) else "Last Dwell on Picture"
            plt.plot(tt_rel[m2], x[m2], color='orange', lw=2.0, label=lab)

    plt.xlabel("Time (ms) relative to picture onset")
    plt.ylabel("X Position")
    plt.title(f"X Position (Participant {participant_number}, Trial {trial_number})")
    plt.legend(loc="best"); plt.tight_layout(); plt.show()


