"""
Cue Task — Pupil size quick visuals (no cheater filter)

Includes:
  • Plot a single trial (raw trace, onset @ 0 ms)
  • Grand average over time (participant-mean, 95% CI)
  • Overlay all traces per condition (Positive / Negative cued picture)
"""

import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from scipy.stats import t
import matplotlib.cm as cm

# ==============================
# Paths & load
# ==============================
ROOT = r"C:/Users/charl/OneDrive/Dokumente/Uni/7. Semester/Bachelor Thesis/ArrowTask"
MAT  = lambda name: f"{ROOT}/{name}"

dominantEye = sio.loadmat(MAT("dominantEye_clean.mat"))["dominantEye"]          
pR = sio.loadmat(MAT("pupilSizeRightPic_clean.mat"))["pupilSizeRight"]          
pL = sio.loadmat(MAT("pupilSizeLeftPic_clean.mat"))["pupilSizeLeft"]            
tV = sio.loadmat(MAT("timeVectorPic_clean.mat"))["timeVector"]                  
cueSide = sio.loadmat(MAT("cueSide_clean.mat"))["cueSide"]                      
pictureSequence = sio.loadmat(MAT("pictureSequence_clean.mat"))["pictureSequence"]  
RT = sio.loadmat(MAT("reactionTime_clean.mat"))["reactionTime"]                 

# ==============================
# preparation
# ==============================
P, T, S = pR.shape
picture_onset_idx = 500
time_ms = tV[0, 0, :] - tV[0, 0, picture_onset_idx]

# dominant-eye selection
Pupil = np.empty_like(pR, dtype=float)
for p in range(P):
    Pupil[p] = pR[p] if int(dominantEye[0, p]) == 0 else pL[p]
Pupil[(Pupil == 0) | (Pupil == 32700)] = np.nan

# ==============================
# Trial grouping (valence of cued picture)
#   rule: picture ID < 44 → positive, else negative
# ==============================
pos_trials, neg_trials = [], []
pos_pids,   neg_pids   = [], []

for p in range(P):
    for tr in range(T):
        cs = cueSide[tr, p]
        if not np.isfinite(cs): 
            continue
        cs = int(cs)
        if cs not in (0, 1):
            continue
        pic_id = pictureSequence[tr, cs, p]
        if not np.isfinite(pic_id):
            continue

        y = Pupil[p, tr, :]
        if y.size != S:
            continue

        if pic_id < 44:
            pos_trials.append(y)
            pos_pids.append(p)
        else:
            neg_trials.append(y)
            neg_pids.append(p)

pos_trials = np.asarray(pos_trials, dtype=float)
neg_trials = np.asarray(neg_trials, dtype=float)
pos_pids   = np.asarray(pos_pids,   dtype=int)
neg_pids   = np.asarray(neg_pids,   dtype=int)

# ==============================
# Helpers
# ==============================
def mean_sem_ci_by_participant(trials, pids, conf=0.95):
    uniq = np.unique(pids)
    pm = []
    for pid in uniq:
        x = trials[pids == pid]
        if x.size:
            pm.append(np.nanmean(x, axis=0))
    pm = np.asarray(pm, float)
    if pm.size == 0:
        return np.full(S, np.nan), np.full(S, np.nan), np.full(S, np.nan), 0
    mean = np.nanmean(pm, axis=0)
    sem  = np.nanstd(pm, axis=0, ddof=1) / np.sqrt(pm.shape[0]/2)
    tcrit = t.ppf((1+conf)/2, df=max(pm.shape[0]-1, 1))
    low, high = mean - tcrit*sem, mean + tcrit*sem
    return mean, low, high, pm.shape[0]

def get_trial(p_idx, tr_idx):
    y = Pupil[p_idx, tr_idx, :].astype(float)
    t = tV[p_idx, tr_idx, :].astype(float)
    return t - t[picture_onset_idx], y


# ==============================
# 1) Plot a single raw trial
# ==============================
def get_trial_blocked(participant, block, trial):
    """Return (t_zeroed, pupil, rt_ms) for participant, block (0/1), trial."""
    gblk = participant * 2 + block
    y = Pupil[gblk, trial, :].astype(float)
    t = tV[gblk, trial, :].astype(float)
    rt_sec = RT[trial, participant] if RT.shape[0] >= trial+1 else np.nan  
    rt_ms = np.nan
    if np.isfinite(rt_sec):
        rt_idx = picture_onset_idx + int(rt_sec * 500) 
        if rt_idx < y.size:
            rt_ms = t[rt_idx] - t[picture_onset_idx]
    return t - t[picture_onset_idx], y, rt_ms

def plot_single_trial(participant=5, block=0, trial=20, title_prefix="Pupil Size over Time"):
    t, y, rt_ms = get_trial_blocked(participant, block, trial)
    plt.figure(figsize=(10, 4.6))
    plt.plot(t, y, lw=1.2)
    plt.axvline(0,   color='black', ls='--', label='Picture onset')
    plt.axvline(500, color='blue',  ls='--', label='LMM timepoint')
    if np.isfinite(rt_ms):
        plt.axvline(rt_ms, color='red', ls='--', label=f'Reaction time ({rt_ms:.0f} ms)')
    plt.xlabel("Time (ms) from picture onset"); plt.ylabel("Pupil size (A.U.)")
    plt.title(f"{title_prefix} — Participant {participant+1}, Block {block+1}, Trial {trial+1}")
    plt.legend(); plt.tight_layout(); plt.show()

# example calls
plot_single_trial(5, 0, 20)
plot_single_trial(12, 1, 12)



# ==============================
# 2) Grand average (participant-mean) + 95% CI
# ==============================
if pos_trials.size and neg_trials.size:
    all_trials = np.vstack([pos_trials, neg_trials])
    all_ids    = np.concatenate([pos_pids,   neg_pids])
elif pos_trials.size:
    all_trials = pos_trials
    all_ids    = pos_pids
else:
    all_trials = neg_trials
    all_ids    = neg_pids

g_mean, g_lo, g_hi, n_subj = mean_sem_ci_by_participant(all_trials, all_ids, conf=0.95)

plt.figure(figsize=(10, 5))
plt.plot(time_ms, g_mean, lw=2, label=f"Mean pupil size")
plt.fill_between(time_ms, g_lo, g_hi, alpha=0.2, label="95% CI")
plt.axvline(0,   color='black', ls='--', label='Picture onset')
plt.axvline(500, color='blue',  ls='--', label='Timepoint for LMM')
plt.xlabel("Time (ms) from picture onset"); plt.ylabel("Pupil size (A.U.)")
plt.title("Average Pupil Size over Time")
plt.legend()
plt.tight_layout()
plt.show()

# ==============================
# 3) Overlay all traces per condition
# ==============================

def plot_all_traces(trials, title, cmap):
    plt.figure(figsize=(12, 5))
    if trials.size == 0:
        plt.title(title + " (no trials)"); plt.tight_layout(); plt.show(); return
    cols = cmap(np.linspace(0, 1, trials.shape[0]))
    for i, tr in enumerate(trials):
        n = min(tr.size, time_ms.size)
        plt.plot(time_ms[:n], tr[:n], color=cols[i], alpha=0.5, lw=0.6)
    plt.axvline(0, color='black', ls='--', label='Picture onset')
    plt.xlabel("Time (ms)")
    plt.ylabel("Pupil size (A.U.)") 
    plt.title(title)
    plt.tight_layout() 
    plt.show()


if pos_trials.size > 0:
    plot_all_traces(pos_trials, "All Pupil Size Curves — Positive Valence Trials (Picture Onset)", cm.viridis)
if neg_trials.size > 0:
    plot_all_traces(neg_trials, "All Pupil Size Curves — Negative Valence Trials (Picture Onset)", cm.plasma)

