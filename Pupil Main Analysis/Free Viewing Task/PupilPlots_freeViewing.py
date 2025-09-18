import numpy as np
import pandas as pd
import scipy.io as sio
import matplotlib.pyplot as plt
from scipy.stats import t

# ==============================
# Load data (your paths)
# ==============================
ROOT = r"C:/Users/charl/OneDrive/Dokumente/Uni/7. Semester/Bachelor Thesis/Eye Data Experiment1Task2/Experiment1_Task2"

dominantEye = pd.read_csv(f"{ROOT}/dominantEye.csv", header=None).values.flatten()
valenceLeft  = pd.read_csv(f"{ROOT}/valenceLeftEye.csv",  header=None).values
valenceRight = pd.read_csv(f"{ROOT}/valenceRightEye.csv", header=None).values
pupilR = sio.loadmat(f"{ROOT}/PicturePupilRight_epoched.mat")["PupilSizeRight"]
pupilL = sio.loadmat(f"{ROOT}/PicturePupilLeft_epoched.mat")["PupilSizeLeft"]
timeV  = sio.loadmat(f"{ROOT}/PictureTimeVector_epoched.mat")["timeVector"]

# ==============================
# Basic prep
# ==============================
participants_to_remove = [42, 43, 120, 121]
dominantEye = np.delete(dominantEye, participants_to_remove, axis=0)
pupilL = pupilL[:, 4:, :]        # drop 4 test trials
pupilR = pupilR[:, 4:, :]
timeV  = timeV[:, 4:, :]

P_blocks, T, S = pupilL.shape        
picture_onset_idx = 500
time_ms = timeV[0,0,:] - timeV[0,0,picture_onset_idx]

# ==============================
# Helpers
# ==============================
def mean_sem_ci_by_participant(trials, part_ids, conf=0.95):
    trials = np.asarray(trials, float)
    part_ids = np.asarray(part_ids, int)
    uniq = np.unique(part_ids)
    pm = []
    for pid in uniq:
        x = trials[part_ids == pid]
        if x.size > 0:
            pm.append(np.nanmean(x, axis=0))
    pm = np.asarray(pm, float)
    mean = np.nanmean(pm, axis=0)
    sem = np.nanstd(pm, axis=0, ddof=1) / np.sqrt(len(pm)/2)
    tcrit = t.ppf((1+conf)/2, df=len(pm)-1)
    low, high = mean - tcrit*sem, mean + tcrit*sem
    return mean, low, high, len(pm)


def append_trials(block_idx, trials_idx, arr_list, pid_list, pid):
    """Append selected trials (mask invalid codes to NaN)."""
    dom = dominantEye[block_idx]
    pup_block = pupilL[block_idx] if dom == 0 else pupilR[block_idx]
    x = np.asarray(pup_block[trials_idx], float)
    x[(x == 0) | (x == 32700)] = np.nan
    arr_list.append(x)
    pid_list.extend([pid] * x.shape[0])

# Map block index -> participant id (0-based after exclusions)
block_to_part = np.repeat(np.arange(P_blocks//2), 2)

# ==============================
# Build labels: current valence, previous valence
# ==============================

val_cur = np.full((P_blocks, T), np.nan)
val_prev = np.full((P_blocks, T), np.nan)

for b in range(P_blocks):
    subj = b // 2
    dom = dominantEye[b]
    # pick valence source for dominant eye
    v_src = valenceLeft if dom == 0 else valenceRight
    # which half (block) of the 176 trials
    base = 0 if (b % 2 == 0) else 88
    # map pupil trials
    for t_idx in range(T):
        orig = base + 4 + t_idx           
        prev_orig = orig - 1
        if 0 <= orig < base+88:
            val_cur[b, t_idx] = v_src[subj, orig]
        if 0 <= prev_orig < base+88:
            val_prev[b, t_idx] = v_src[subj, prev_orig]

# ==============================
# GROUPS
# ==============================
# 1) Current valence: Positive(1) vs Negative(0)
pos_cur, pos_pid = [], []
neg_cur, neg_pid = [], []

# 2) Previous valence
pos_prev, pos_prev_pid = [], []
neg_prev, neg_prev_pid = [], []

# 3) Interaction (prev x curr): (0,0), (0,1), (1,0), (1,1)
g00, g00_pid = [], []
g01, g01_pid = [], []
g10, g10_pid = [], []
g11, g11_pid = [], []

for b in range(P_blocks):
    pid = block_to_part[b]
    # mask of valid length traces
    valid_mask = np.isfinite(val_cur[b]) & (np.asarray(pupilL[b] if dominantEye[b]==0 else pupilR[b]).shape[1] == S)
    # CURRENT
    cur_pos_idx = np.where(valid_mask & (val_cur[b] == 1))[0]
    cur_neg_idx = np.where(valid_mask & (val_cur[b] == 0))[0]
    if cur_pos_idx.size: append_trials(b, cur_pos_idx, pos_cur, pos_pid, pid)
    if cur_neg_idx.size: append_trials(b, cur_neg_idx, neg_cur, neg_pid, pid)
    # PREVIOUS (skip where previous is NaN, e.g., the first usable trial in a block)
    prev_pos_idx = np.where(valid_mask & (val_prev[b] == 1))[0]
    prev_neg_idx = np.where(valid_mask & (val_prev[b] == 0))[0]
    if prev_pos_idx.size: append_trials(b, prev_pos_idx, pos_prev, pos_prev_pid, pid)
    if prev_neg_idx.size: append_trials(b, prev_neg_idx, neg_prev, neg_prev_pid, pid)
    # INTERACTION
    idx_00 = np.where(valid_mask & (val_prev[b] == 0) & (val_cur[b] == 0))[0]
    idx_01 = np.where(valid_mask & (val_prev[b] == 0) & (val_cur[b] == 1))[0]
    idx_10 = np.where(valid_mask & (val_prev[b] == 1) & (val_cur[b] == 0))[0]
    idx_11 = np.where(valid_mask & (val_prev[b] == 1) & (val_cur[b] == 1))[0]
    if idx_00.size: append_trials(b, idx_00, g00, g00_pid, pid)
    if idx_01.size: append_trials(b, idx_01, g01, g01_pid, pid)
    if idx_10.size: append_trials(b, idx_10, g10, g10_pid, pid)
    if idx_11.size: append_trials(b, idx_11, g11, g11_pid, pid)

# Stack
pos_cur = np.vstack(pos_cur) if len(pos_cur) else np.empty((0, S))
neg_cur = np.vstack(neg_cur) if len(neg_cur) else np.empty((0, S))
pos_prev = np.vstack(pos_prev) if len(pos_prev) else np.empty((0, S))
neg_prev = np.vstack(neg_prev) if len(neg_prev) else np.empty((0, S))
g00 = np.vstack(g00) if len(g00) else np.empty((0, S))
g01 = np.vstack(g01) if len(g01) else np.empty((0, S))
g10 = np.vstack(g10) if len(g10) else np.empty((0, S))
g11 = np.vstack(g11) if len(g11) else np.empty((0, S))

# ==============================
# helpers for calculating descriptive stats of pupil size at 500 ms post picture onset
# ==============================
from scipy.stats import t

LMM_IDX = 750

def participant_means_at(trials, pids, idx=LMM_IDX):
    trials = np.asarray(trials, float)
    pids = np.asarray(pids, int)
    vals = []
    for pid in np.unique(pids):
        x = trials[pids == pid, idx]
        if x.size:
            vals.append(np.nanmean(x))
    return np.asarray(vals, float)

def describe_vals(v):
    v = v[~np.isnan(v)]
    n = v.size
    mean = float(np.nanmean(v)) if n else np.nan
    sd   = float(np.nanstd(v, ddof=1)) if n > 1 else np.nan
    vmin = float(np.nanmin(v)) if n else np.nan
    vmax = float(np.nanmax(v)) if n else np.nan
    if n > 1:
        sem = sd / np.sqrt(n)
        tcrit = t.ppf(0.975, df=n-1)
        lo, hi = mean - tcrit*sem, mean + tcrit*sem
    else:
        lo = hi = np.nan
    return dict(n=int(n), mean=mean, sd=sd, min=vmin, max=vmax, lo=lo, hi=hi)


# ==============================
# Plot: Current valence
# ==============================
m_pos, lo_pos, hi_pos, n_pos = mean_sem_ci_by_participant(pos_cur, np.array(pos_pid))
m_neg, lo_neg, hi_neg, n_neg = mean_sem_ci_by_participant(neg_cur, np.array(neg_pid))

plt.figure(figsize=(10, 5))
plt.plot(time_ms, m_pos, label=f'Positive', linewidth=1.8)
plt.fill_between(time_ms, lo_pos, hi_pos, alpha=0.2)
plt.plot(time_ms, m_neg, label=f'Negative', linewidth=1.8)
plt.fill_between(time_ms, lo_neg, hi_neg, alpha=0.2)
plt.axvline(0, color='black', ls='--', label='Picture Onset')
plt.axvline(500, color='tab:blue', ls='--', label='Timepoint for LMM')
plt.xlabel("Time (ms) from Picture Onset"); plt.ylabel("Pupil size in A.U.")
plt.title("Pupil Response by Valence of first looked Picture")
plt.legend(); plt.tight_layout(); plt.show()


pos_v = participant_means_at(pos_cur, np.array(pos_pid), idx=LMM_IDX)
neg_v = participant_means_at(neg_cur, np.array(neg_pid), idx=LMM_IDX)

pos_stats = describe_vals(pos_v)
neg_stats = describe_vals(neg_v)

print(f"\n=== LMM @ {LMM_IDX} (Current Valence; participant-level) ===")
print(f"Positive: n={pos_stats['n']}, mean={pos_stats['mean']:.4f}, sd={pos_stats['sd']:.4f}, "
      f"min={pos_stats['min']:.4f}, max={pos_stats['max']:.4f}, "
      f"95% CI=({pos_stats['lo']:.4f}, {pos_stats['hi']:.4f})")
print(f"Negative: n={neg_stats['n']}, mean={neg_stats['mean']:.4f}, sd={neg_stats['sd']:.4f}, "
      f"min={neg_stats['min']:.4f}, max={neg_stats['max']:.4f}, "
      f"95% CI=({neg_stats['lo']:.4f}, {neg_stats['hi']:.4f})")


# ==============================
# Plot: Previous valence
# ==============================
m_pv, lo_pv, hi_pv, n_pv = mean_sem_ci_by_participant(pos_prev, np.array(pos_prev_pid))
m_nv, lo_nv, hi_nv, n_nv = mean_sem_ci_by_participant(neg_prev, np.array(neg_prev_pid))

plt.figure(figsize=(10, 5))
plt.plot(time_ms, m_pv, label=f'Positive', linewidth=1.8)
plt.fill_between(time_ms, lo_pv, hi_pv, alpha=0.2)
plt.plot(time_ms, m_nv, label=f'Negative', linewidth=1.8)
plt.fill_between(time_ms, lo_nv, hi_nv, alpha=0.2)
plt.axvline(0, color='black', ls='--'); plt.axvline(500, color='tab:blue', ls='--')
plt.xlabel("Time in ms (relative to picture onset)"); plt.ylabel("Pupil size in A.U.)")
plt.title("Average Pupil Size by Previous Valence")
plt.legend(); plt.tight_layout(); plt.show()


pv_v = participant_means_at(pos_prev, np.array(pos_prev_pid), idx=LMM_IDX)
nv_v = participant_means_at(neg_prev, np.array(neg_prev_pid), idx=LMM_IDX)

pv_stats = describe_vals(pv_v)
nv_stats = describe_vals(nv_v)

print(f"\n=== LMM @ {LMM_IDX} (Previous Valence; participant-level) ===")
print(f"Prev Positive: n={pv_stats['n']}, mean={pv_stats['mean']:.4f}, sd={pv_stats['sd']:.4f}, "
      f"min={pv_stats['min']:.4f}, max={pv_stats['max']:.4f}, "
      f"95% CI=({pv_stats['lo']:.4f}, {pv_stats['hi']:.4f})")
print(f"Prev Negative: n={nv_stats['n']}, mean={nv_stats['mean']:.4f}, sd={nv_stats['sd']:.4f}, "
      f"min={nv_stats['min']:.4f}, max={nv_stats['max']:.4f}, "
      f"95% CI=({nv_stats['lo']:.4f}, {nv_stats['hi']:.4f})")


# ==============================
# Plot: Interaction (Prev x Current)
# ==============================
m00, lo00, hi00, n00 = mean_sem_ci_by_participant(g00, np.array(g00_pid))  # prev=0, cur=0
m01, lo01, hi01, n01 = mean_sem_ci_by_participant(g01, np.array(g01_pid))  # prev=0, cur=1
m10, lo10, hi10, n10 = mean_sem_ci_by_participant(g10, np.array(g10_pid))  # prev=1, cur=0
m11, lo11, hi11, n11 = mean_sem_ci_by_participant(g11, np.array(g11_pid))  # prev=1, cur=1

plt.figure(figsize=(10, 5))
plt.plot(time_ms, m00, linewidth=1.6, label=f"Negative --> Negative")
plt.fill_between(time_ms, lo00, hi00, alpha=0.15)
plt.plot(time_ms, m01, linewidth=1.6, label=f"Negative --> Positive")
plt.fill_between(time_ms, lo01, hi01, alpha=0.15)
plt.plot(time_ms, m10, linewidth=1.6, label=f"Positive --> Negative")
plt.fill_between(time_ms, lo10, hi10, alpha=0.15)
plt.plot(time_ms, m11, linewidth=1.6, label=f"Positive --> Positive")
plt.fill_between(time_ms, lo11, hi11, alpha=0.15)
plt.axvline(0, color='black', ls='--'); plt.axvline(500, color='tab:blue', ls='--')
plt.xlabel("Time in ms (relative to picture onset)"); plt.ylabel("Pupil size in A.U.")
plt.title("Average Pupil Size by Valence and Previous Valence")
plt.legend(ncol=2); plt.tight_layout(); plt.show()


v00 = participant_means_at(g00, np.array(g00_pid), idx=LMM_IDX)  # Neg→Neg
v01 = participant_means_at(g01, np.array(g01_pid), idx=LMM_IDX)  # Neg→Pos
v10 = participant_means_at(g10, np.array(g10_pid), idx=LMM_IDX)  # Pos→Neg
v11 = participant_means_at(g11, np.array(g11_pid), idx=LMM_IDX)  # Pos→Pos

s00 = describe_vals(v00); s01 = describe_vals(v01)
s10 = describe_vals(v10); s11 = describe_vals(v11)

print(f"\n=== LMM @ {LMM_IDX} (Prev × Current; participant-level) ===")
print(f"Neg → Neg: n={s00['n']}, mean={s00['mean']:.4f}, sd={s00['sd']:.4f}, "
      f"min={s00['min']:.4f}, max={s00['max']:.4f}, 95% CI=({s00['lo']:.4f}, {s00['hi']:.4f})")
print(f"Neg → Pos: n={s01['n']}, mean={s01['mean']:.4f}, sd={s01['sd']:.4f}, "
      f"min={s01['min']:.4f}, max={s01['max']:.4f}, 95% CI=({s01['lo']:.4f}, {s01['hi']:.4f})")
print(f"Pos → Neg: n={s10['n']}, mean={s10['mean']:.4f}, sd={s10['sd']:.4f}, "
      f"min={s10['min']:.4f}, max={s10['max']:.4f}, 95% CI=({s10['lo']:.4f}, {s10['hi']:.4f})")
print(f"Pos → Pos: n={s11['n']}, mean={s11['mean']:.4f}, sd={s11['sd']:.4f}, "
      f"min={s11['min']:.4f}, max={s11['max']:.4f}, 95% CI=({s11['lo']:.4f}, {s11['hi']:.4f})")
