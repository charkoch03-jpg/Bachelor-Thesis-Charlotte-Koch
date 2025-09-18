import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from scipy.stats import t

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
blockOrder = sio.loadmat(MAT("blockOrder_clean.mat"))["blockOrder"]            
pictureSequence = sio.loadmat(MAT("pictureSequence_clean.mat"))["pictureSequence"]  
cheater = sio.loadmat(MAT("cheater.mat"))["cheater"] 
cheater_mask = (cheater == 0)  # True for non cheater trials

# ==============================
# Basic prep
# ==============================
B, TR, S = pR.shape                       
P = B // 2                                
picture_onset_idx = 500
time_ms = tV[0,0,:] - tV[0,0,picture_onset_idx]

# map block -> participant id
block_to_part = np.repeat(np.arange(P), 2)

# ==============================
# Helper: choose dominant-eye pupil for a given block
# ==============================
def get_block_pupil(block):
    subj = block // 2
    dom = int(dominantEye[0, subj])
    return pR if dom == 0 else pL

# ==============================
# Derive per-trial labels: current valence, previous valence, task condition
# ==============================
val_cur = np.full((B, TR), np.nan)     
val_prev = np.full((B, TR), np.nan)     
task = np.empty((B, TR), dtype=object)  

for b in range(B):
    order = str(blockOrder[0, b]).strip()  
    first_half_is_congruent = (order == 'A')
    for t in range(TR):
        task[b, t] = 'congruent' if (t<44 and first_half_is_congruent) else \
                     'incongruent' if (t<44 and not first_half_is_congruent) else \
                     'incongruent' if (t>=44 and first_half_is_congruent) else 'congruent'

        side = int(cueSide[t, b])              
        pic_idx = int(pictureSequence[t, side, b]) 
        val_cur[b, t] = 1 if pic_idx < 44 else 0

    vp = val_cur[b]
    val_prev[b, 1:] = vp[:-1]  # first trial remains NaN

# ==============================
# Helper functions
# ==============================
def collect_trials(mask):
    xs, pids = [], []
    for b in range(B):
        idx = np.where(mask[b])[0]
        if idx.size == 0: 
            continue
        Pblk = get_block_pupil(b)[b] 
        X = np.asarray(Pblk[idx], float)
        X[(X == 0) | (X == 32700)] = np.nan
        xs.append(X)
        pids.extend([block_to_part[b]] * X.shape[0])
    if len(xs) == 0:
        return np.empty((0, S), float), np.array([], int)
    return np.vstack(xs), np.array(pids, int)

def mean_sem_ci_by_participant(trials, pids, conf=0.95):
    trials = np.asarray(trials, float)
    pids = np.asarray(pids, int)
    uniq = np.unique(pids)
    pm = []
    for pid in uniq:
        x = trials[pids == pid]
        if x.size:
            pm.append(np.nanmean(x, axis=0))
    pm = np.asarray(pm, float)
    if pm.size == 0:
        return (np.full(S, np.nan),)*4
    mean = np.nanmean(pm, axis=0)
    sd   = np.nanstd(pm, axis=0, ddof=1)
    n    = pm.shape[0]/2
    sem  = sd / np.sqrt(n)
    tcrit = t.ppf((1+conf)/2, df=max(n-1,1))
    lo, hi = mean - tcrit*sem, mean + tcrit*sem
    return mean, lo, hi, n

LMM_IDX = 750

def participant_means_at(trials, pids, idx=LMM_IDX):
    pids = np.asarray(pids, int)
    trials = np.asarray(trials, float)
    vals = []
    for pid in np.unique(pids):
        x = trials[pids == pid, idx]
        x = x[np.isfinite(x)]
        if x.size:
            vals.append(np.nanmean(x))
    return np.asarray(vals, float)


def finish_plot(title):
    plt.axvline(0,   color='black', ls='--', label='Picture onset')
    plt.xlabel("Time (ms) relative to Picture Onset")
    plt.ylabel("Average Pupil Size")
    plt.title(title)
    plt.legend()
    plt.grid(True, ls='--', alpha=0.3)
    plt.tight_layout()
    plt.show()

# ==============================
# Plotting (only cheater trials)
# ==============================

# Current valence
mask_pos = (val_cur == 1) & cheater_mask
mask_neg = (val_cur == 0) & cheater_mask
pos_trials, pos_pids = collect_trials(mask_pos)
neg_trials, neg_pids = collect_trials(mask_neg)
m_pos, lo_pos, hi_pos, n_pos = mean_sem_ci_by_participant(pos_trials, pos_pids)
m_neg, lo_neg, hi_neg, n_neg = mean_sem_ci_by_participant(neg_trials, neg_pids)
plt.figure(figsize=(10,5))
plt.plot(time_ms, m_pos, label='Positive', lw=1.8, color='green')
plt.fill_between(time_ms, lo_pos, hi_pos, alpha=0.2, color='green')
plt.plot(time_ms, m_neg, label='Negative', lw=1.8, color='red')
plt.fill_between(time_ms, lo_neg, hi_neg, alpha=0.2, color='red')
finish_plot("Average Pupil Size over Time for non-Cheaters by Current Valence")

# Previous valence
mask_prev_pos = (val_prev == 1) & cheater_mask
mask_prev_neg = (val_prev == 0) & cheater_mask
pp_trials, pp_pids = collect_trials(mask_prev_pos)
pn_trials, pn_pids = collect_trials(mask_prev_neg)
m_pp, lo_pp, hi_pp, n_pp = mean_sem_ci_by_participant(pp_trials, pp_pids)
m_pn, lo_pn, hi_pn, n_pn = mean_sem_ci_by_participant(pn_trials, pn_pids)
plt.figure(figsize=(10,5))
plt.plot(time_ms, m_pp, label='Positive', lw=1.8, color='green')
plt.fill_between(time_ms, lo_pp, hi_pp, alpha=0.2, color='green')
plt.plot(time_ms, m_pn, label='Negative', lw=1.8, color='red')
plt.fill_between(time_ms, lo_pn, hi_pn, alpha=0.2, color='red')
finish_plot("Average Pupil Size over Time for non-Cheaters by Previous Valence)")


# Task condition
mask_cong = (task=='congruent') & cheater_mask
mask_incg = (task=='incongruent') & cheater_mask
cg_trials, cg_pids = collect_trials(mask_cong)
ic_trials, ic_pids = collect_trials(mask_incg)
m_cg, lo_cg, hi_cg, n_cg = mean_sem_ci_by_participant(cg_trials, cg_pids)
m_ic, lo_ic, hi_ic, n_ic = mean_sem_ci_by_participant(ic_trials, ic_pids)
plt.figure(figsize=(10,5))
plt.plot(time_ms, m_cg, label='Congruent', lw=1.8, color='purple')
plt.fill_between(time_ms, lo_cg, hi_cg, alpha=0.2, color='purple')
plt.plot(time_ms, m_ic, label='Incongruent', lw=1.8, color='orange')
plt.fill_between(time_ms, lo_ic, hi_ic, alpha=0.2, color='orange')
finish_plot("Average Pupil Size over Time for non-Cheaters by Task Condition")

