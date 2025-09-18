"""
Reaction-time summary — Cue/Arrow task

Loads reactionTimes_clean.mat and reports:
  • N valid trials
  • Mean (s), SD (s), Min (s), Max (s)
"""

import numpy as np
import scipy.io as sio

# === Paths (your structure) ===
ROOT = r"C:/Users/charl/OneDrive/Dokumente/Uni/7. Semester/Bachelor Thesis/ArrowTask"
MAT_RT_CLEAN = f"{ROOT}/reactionTimes_clean.mat"   # variable name: 'reactionTimes'

# === Load cleaned RTs ===
mat = sio.loadmat(MAT_RT_CLEAN)
RT = mat["reactionTimes"]  # shape: (trials, blocks), with NaNs for invalid

# === Flatten and drop NaNs ===
rt_flat = RT[~np.isnan(RT)].astype(float)

# === Compute stats ===
if rt_flat.size > 0:
    N = rt_flat.size
    mean_rt = np.mean(rt_flat)
    sd_rt   = np.std(rt_flat, ddof=1) if rt_flat.size > 1 else np.nan
    min_rt  = np.min(rt_flat)
    max_rt  = np.max(rt_flat)

    print("=== Overall RT summary (seconds) ===")
    print(f"N trials: {N}")
    print(f"Mean RT: {mean_rt:.3f} s")
    print(f"SD RT:   {sd_rt:.3f} s")
    print(f"Min RT:  {min_rt:.3f} s")
    print(f"Max RT:  {max_rt:.3f} s")
else:
    print("⚠️ No valid reaction times found.")
