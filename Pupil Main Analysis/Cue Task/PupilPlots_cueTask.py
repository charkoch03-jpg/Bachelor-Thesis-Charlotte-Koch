"""
Cue Task — Average pupil size plots + LMM timepoint stats

Includes picture-locked:
  • Current valence (pos/neg)
  • Previous valence (pos/neg)
  • Interaction: Previous × Current (4 curves)
  • Task condition (congruent/incongruent)
  • Interaction: Task × Current valence (4 curves)
  • For each plot: participant-level mean curve with 95% CI
  • LMM timepoint (index 750) stats computed on participant means
"""

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
    # first 44 trials congruent in A, incongruent in B
    first_half_is_congruent = (order == 'A')
    for t in range(TR):
        # task condition
        if t < 44:
            task[b, t] = 'congruent' if first_half_is_congruent else 'incongruent'
        else:
            task[b, t] = 'incongruent' if first_half_is_congruent else 'congruent'

        # current valence
        side = int(cueSide[t, b])              
        pic_idx = int(pictureSequence[t, side, b]) 
        val_cur[b, t] = 1 if pic_idx < 44 else 0

    # previous valence 
    vp = val_cur[b]
    val_prev[b, 1:] = vp[:-1]                  # first trial remains NaN

# ==============================
# Gather trial data into groups
# ==============================
def collect_trials(mask):
    """Return trials [Ntrials x S] and participant IDs [Ntrials] for mask over (B,TR)."""
    xs, pids = [], []
    for b in range(B):
        idx = np.where(mask[b])[0]
        if idx.size == 0: 
            continue
        Pblk = get_block_pupil(b)[b]          
        # copy + clean codes to NaN
        X = np.asarray(Pblk[idx], float)
        X[(X == 0) | (X == 32700)] = np.nan
        xs.append(X)
        pids.extend([block_to_part[b]] * X.shape[0])
    if len(xs) == 0:
        return np.empty((0, S), float), np.array([], int)
    return np.vstack(xs), np.array(pids, int)

def mean_sem_ci_by_participant(trials, pids, conf=0.95):
    """Per-participant means → grand mean and CI."""
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
    """Participant-level averages at a single time index."""
    pids = np.asarray(pids, int)
    trials = np.asarray(trials, float)
    vals = []
    for pid in np.unique(pids):
        x = trials[pids == pid, idx]
        x = x[np.isfinite(x)]
        if x.size:
            vals.append(np.nanmean(x))
    return np.asarray(vals, float)

def describe_vals(v):
    v = v[np.isfinite(v)]
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

def finish_plot(title):
    plt.axvline(0,   color='black', ls='--', label='Picture onset')
    plt.axvline(500, color='tab:blue', ls='--', label='Timepoint for LMM')
    plt.xlabel("Time (ms) from picture onset")
    plt.ylabel("Pupil size in A.U.")
    plt.title(title)
    plt.legend()
    plt.grid(True, ls='--', alpha=0.3)
    plt.tight_layout()
    plt.show()

# ==============================
# Plot 1: Current valence
# ==============================
mask_pos = (val_cur == 1)
mask_neg = (val_cur == 0)

pos_trials, pos_pids = collect_trials(mask_pos)
neg_trials, neg_pids = collect_trials(mask_neg)

m_pos, lo_pos, hi_pos, n_pos = mean_sem_ci_by_participant(pos_trials, pos_pids)
m_neg, lo_neg, hi_neg, n_neg = mean_sem_ci_by_participant(neg_trials, neg_pids)

plt.figure(figsize=(10,5))
plt.plot(time_ms, m_pos, label=f'Positive', lw=1.8, color='green')
plt.fill_between(time_ms, lo_pos, hi_pos, alpha=0.2, color='green')
plt.plot(time_ms, m_neg, label=f'Negative', lw=1.8, color='red')
plt.fill_between(time_ms, lo_neg, hi_neg, alpha=0.2, color='red')
finish_plot("Pupil Response by Valence of first looked picture")

# stats @ 750
pv = participant_means_at(pos_trials, pos_pids, LMM_IDX)
nv = participant_means_at(neg_trials, neg_pids, LMM_IDX)
ps = describe_vals(pv); ns = describe_vals(nv)
print(f"\n=== LMM @ {LMM_IDX} (Current Valence; participant-level) ===")
print(f"Positive: n={ps['n']}, mean={ps['mean']:.4f}, sd={ps['sd']:.4f}, "
      f"min={ps['min']:.4f}, max={ps['max']:.4f}, 95% CI=({ps['lo']:.4f}, {ps['hi']:.4f})")
print(f"Negative: n={ns['n']}, mean={ns['mean']:.4f}, sd={ns['sd']:.4f}, "
      f"min={ns['min']:.4f}, max={ns['max']:.4f}, 95% CI=({ns['lo']:.4f}, {ns['hi']:.4f})")

# ==============================
# Plot 2: Previous valence
# ==============================
mask_prev_pos = (val_prev == 1)
mask_prev_neg = (val_prev == 0)

pp_trials, pp_pids = collect_trials(mask_prev_pos)
pn_trials, pn_pids = collect_trials(mask_prev_neg)

m_pp, lo_pp, hi_pp, n_pp = mean_sem_ci_by_participant(pp_trials, pp_pids)
m_pn, lo_pn, hi_pn, n_pn = mean_sem_ci_by_participant(pn_trials, pn_pids)

plt.figure(figsize=(10,5))
plt.plot(time_ms, m_pp, label='Positive', lw=1.8, color='green')
plt.fill_between(time_ms, lo_pp, hi_pp, alpha=0.2, color='green')
plt.plot(time_ms, m_pn, label='Negative', lw=1.8, color='red')
plt.fill_between(time_ms, lo_pn, hi_pn, alpha=0.2, color='red')
finish_plot("Pupil Response by Previous Valence")

pp_v = participant_means_at(pp_trials, pp_pids, LMM_IDX)
pn_v = participant_means_at(pn_trials, pn_pids, LMM_IDX)
pps = describe_vals(pp_v); pns = describe_vals(pn_v)
print(f"\n=== LMM @ {LMM_IDX} (Previous Valence; participant-level) ===")
print(f"Prev Positive: n={pps['n']}, mean={pps['mean']:.4f}, sd={pps['sd']:.4f}, "
      f"min={pps['min']:.4f}, max={pps['max']:.4f}, 95% CI=({pps['lo']:.4f}, {pps['hi']:.4f})")
print(f"Prev Negative: n={pns['n']}, mean={pns['mean']:.4f}, sd={pns['sd']:.4f}, "
      f"min={pns['min']:.4f}, max={pns['max']:.4f}, 95% CI=({pns['lo']:.4f}, {pns['hi']:.4f})")

# ==============================
# Plot 3: Interaction (Prev × Current)
# ==============================
m00_trials, m00_pids = collect_trials((val_prev==0) & (val_cur==0))
m01_trials, m01_pids = collect_trials((val_prev==0) & (val_cur==1))
m10_trials, m10_pids = collect_trials((val_prev==1) & (val_cur==0))
m11_trials, m11_pids = collect_trials((val_prev==1) & (val_cur==1))

m00, lo00, hi00, n00 = mean_sem_ci_by_participant(m00_trials, m00_pids)
m01, lo01, hi01, n01 = mean_sem_ci_by_participant(m01_trials, m01_pids)
m10, lo10, hi10, n10 = mean_sem_ci_by_participant(m10_trials, m10_pids)
m11, lo11, hi11, n11 = mean_sem_ci_by_participant(m11_trials, m11_pids)

plt.figure(figsize=(10,5))
plt.plot(time_ms, m00, lw=1.6, label="Negative → Negative", color='tab:red')
plt.fill_between(time_ms, lo00, hi00, alpha=0.15, color='tab:red')
plt.plot(time_ms, m01, lw=1.6, label="Negative  → Positive", color='tab:orange')
plt.fill_between(time_ms, lo01, hi01, alpha=0.15, color='tab:orange')
plt.plot(time_ms, m10, lw=1.6, label="Positive → Negative", color='tab:blue')
plt.fill_between(time_ms, lo10, hi10, alpha=0.15, color='tab:blue')
plt.plot(time_ms, m11, lw=1.6, label="Positive → Positive", color='tab:green')
plt.fill_between(time_ms, lo11, hi11, alpha=0.15, color='tab:green')
finish_plot("Pupil Response by Valence and Previous Valence")

for name, trials, pids in [
    ("Negative → Negative", m00_trials, m00_pids),
    ("Negative → Positive", m01_trials, m01_pids),
    ("Positive → Negative", m10_trials, m10_pids),
    ("Positive → Positive", m11_trials, m11_pids),
]:
    v = participant_means_at(trials, pids, LMM_IDX)
    s = describe_vals(v)
    print(f"{name:<12} — n={s['n']}, mean={s['mean']:.4f}, sd={s['sd']:.4f}, "
          f"min={s['min']:.4f}, max={s['max']:.4f}, 95% CI=({s['lo']:.4f}, {s['hi']:.4f})")

# ==============================
# Plot 4: Task condition
# ==============================
mask_cong = (task == 'congruent')
mask_incg = (task == 'incongruent')

cg_trials, cg_pids = collect_trials(mask_cong)
ic_trials, ic_pids = collect_trials(mask_incg)

m_cg, lo_cg, hi_cg, n_cg = mean_sem_ci_by_participant(cg_trials, cg_pids)
m_ic, lo_ic, hi_ic, n_ic = mean_sem_ci_by_participant(ic_trials, ic_pids)

plt.figure(figsize=(10,5))
plt.plot(time_ms, m_cg, label=f'Congruent (n={n_cg})', lw=1.8, color='purple')
plt.fill_between(time_ms, lo_cg, hi_cg, alpha=0.2, color='purple')
plt.plot(time_ms, m_ic, label=f'Incongruent (n={n_ic})', lw=1.8, color='orange')
plt.fill_between(time_ms, lo_ic, hi_ic, alpha=0.2, color='orange')
finish_plot("Pupil Response by Task Condition")

cg_v = participant_means_at(cg_trials, cg_pids, LMM_IDX)
ic_v = participant_means_at(ic_trials, ic_pids, LMM_IDX)
cgs = describe_vals(cg_v); ics = describe_vals(ic_v)
print(f"\n=== LMM @ {LMM_IDX} (Task; participant-level) ===")
print(f"Congruent:   n={cgs['n']}, mean={cgs['mean']:.4f}, sd={cgs['sd']:.4f}, "
      f"min={cgs['min']:.4f}, max={cgs['max']:.4f}, 95% CI=({cgs['lo']:.4f}, {cgs['hi']:.4f})")
print(f"Incongruent: n={ics['n']}, mean={ics['mean']:.4f}, sd={ics['sd']:.4f}, "
      f"min={ics['min']:.4f}, max={ics['max']:.4f}, 95% CI=({ics['lo']:.4f}, {ics['hi']:.4f})")

# ==============================
# Plot 5: Task × Current valence (4 curves)
# ==============================
cells_tv = [
    ('Congruent Negative',  (task=='congruent')   & (val_cur==0), 'tab:red'),
    ('Congruent Positive',  (task=='congruent')   & (val_cur==1), 'tab:green'),
    ('Incongruent Negative',(task=='incongruent') & (val_cur==0), 'tab:orange'),
    ('Incongruent Positive',(task=='incongruent') & (val_cur==1), 'tab:blue'),
]

plt.figure(figsize=(10,5))
for label, msk, col in cells_tv:
    tr, pid = collect_trials(msk)
    m, lo, hi, n = mean_sem_ci_by_participant(tr, pid)
    plt.plot(time_ms, m, lw=1.6, label=f"{label} (n={n})", color=col)
    plt.fill_between(time_ms, lo, hi, alpha=0.15, color=col)
finish_plot("Pupil Response for Valence and Task Condition")

print(f"\n=== LMM @ {LMM_IDX} (Task × Current Valence; participant-level) ===")
for label, msk, _ in cells_tv:
    tr, pid = collect_trials(msk)
    v = participant_means_at(tr, pid, LMM_IDX)
    s = describe_vals(v)
    print(f"{label:<26} — n={s['n']}, mean={s['mean']:.4f}, sd={s['sd']:.4f}, "
          f"min={s['min']:.4f}, max={s['max']:.4f}, 95% CI=({s['lo']:.4f}, {s['hi']:.4f})")
