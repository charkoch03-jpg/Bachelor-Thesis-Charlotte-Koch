"""
Cue Task — Cross dwell detection

• Uses default band 960±100 px, with fallback center±100.
• Saves fixation start/end (ms rel. cross onset) + calibration centers (px).
• Plot shows used thresholds (blue) and original defaults (orange if adapted).
"""

import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt

# --- Parameters ---
stability_threshold = 50
start_end_stability_threshold = 50
window_size = 30
min_out_of_threshold_duration = 50

default_center = 960.0
default_lower, default_upper = default_center - 100, default_center + 100
TRIALS_PER_BLOCK = 88

# --- Paths ---
ROOT = r"C:/Users/charl/OneDrive/Dokumente/Uni/7. Semester/Bachelor Thesis/ArrowTask"
MAT = lambda name: f"{ROOT}/{name}"

dominantEye = sio.loadmat(MAT("dominantEye_clean.mat"))["dominantEye"]
xR_obj = sio.loadmat(MAT("xPositionRightCrosstoCross_clean.mat"))["xPositionRight"]
xL_obj = sio.loadmat(MAT("xPositionLeftCrosstoCross_clean.mat"))["xPositionLeft"]
t_obj  = sio.loadmat(MAT("timeVectorCrosstoCross_clean.mat"))["timeVector"]
po_obj = sio.loadmat(MAT("pictureOnsets_idx_CrosstoCross.mat"))["pictureOnsetsIdx"]
co_obj = sio.loadmat(MAT("crossOnsets_idx_CrosstoCross.mat"))["crossOnsetsIdx"]
rt_sec = sio.loadmat(MAT("reactionTimes_clean.mat"))["reactionTimes"]

# --- Dominant eye selection ---
P, T = t_obj.shape
x_obj = np.empty_like(xR_obj, dtype=object)
for p in range(P):
    x_obj[p, :] = xR_obj[p, :] if int(dominantEye[0, p]) == 0 else xL_obj[p, :]

# --- Helpers ---
as_1d = lambda a: np.asarray(a).ravel()
_scalar = lambda cell: float(as_1d(cell)[0]) if as_1d(cell).size else np.nan

def detect_longest_fixation(x, t, lower, upper, end_time=None):
    fixes, start, out = [], None, 0
    for i in range(len(x)):
        if end_time is not None and t[i] >= end_time: break
        if np.isnan(x[i]): continue
        if i + window_size >= len(x): break
        w = x[i:i+window_size]
        if start is None:
            if lower <= np.nanmin(w) and np.nanmax(w) <= upper:
                if (np.nanmax(w) - np.nanmin(w)) <= start_end_stability_threshold:
                    start = t[i+2]
        else:
            if lower <= x[i] <= upper:
                out = 0
            else:
                out += 1
                if out > min_out_of_threshold_duration:
                    fixes.append((start, t[i - min_out_of_threshold_duration]))
                    start, out = None, 0
    if start is not None and end_time is not None:
        fixes.append((start, end_time))
    return max(fixes, key=lambda f: f[1]-f[0]) if fixes else (None, None)

def local_pic_time_ms(p, tr, t_vec):
    pic_g, cross_g = _scalar(po_obj[p, tr]), _scalar(co_obj[p, tr])
    if not (np.isfinite(pic_g) and np.isfinite(cross_g)): return np.nan
    local_samples = pic_g - cross_g
    for per_ms in (2.0, 1.0):
        pt = t_vec[0] + local_samples * per_ms
        if t_vec[0] <= pt <= t_vec[-1]: return pt
    return np.nan

def estimate_center(x, i_end, max_samples=300):
    take = int(min(max_samples, max(0, i_end), len(x)))
    if take <= 5: return np.nan
    seg = x[:take][~np.isnan(x[:take])]
    return float(np.median(seg)) if seg.size >= 5 else np.nan

# --- Detection ---
fix_start = np.full((P, T), np.nan)
fix_end   = np.full((P, T), np.nan)
cal_shift = np.full((P, T), np.nan)

for p in range(P):
    for tr in range(T):
        x = as_1d(x_obj[p, tr]).astype(float)
        t = as_1d(t_obj[p, tr]).astype(float)
        if x.size == 0 or t.size == 0: continue
        pic_time = local_pic_time_ms(p, tr, t)

        s_ms, e_ms = detect_longest_fixation(x, t, default_lower, default_upper, end_time=pic_time)
        center_used = default_center

        if s_ms is None:
            i_end = int(np.searchsorted(t, pic_time) - window_size - 1) if np.isfinite(pic_time) else len(t)
            c_est = estimate_center(x, i_end, 300)
            if np.isfinite(c_est):
                center_used = c_est
                s_ms, e_ms = detect_longest_fixation(x, t, c_est-100, c_est+100, end_time=pic_time)

        if s_ms is not None and e_ms is not None:
            fix_start[p, tr] = s_ms - t[0]
            fix_end[p, tr]   = e_ms - t[0]
            cal_shift[p, tr] = center_used

print("✅ Cue task cross-dwell detection complete.")

# --- Save ---
sio.savemat(MAT("fixationCrossStartTimes_LOCAL.mat"), {"fixationStartTimes": fix_start})
sio.savemat(MAT("fixationCrossEndTimes_LOCAL.mat"),   {"fixationEndTimes":   fix_end})
sio.savemat(MAT("calibrationShifts_LOCAL.mat"),       {"calibrationShifts":  cal_shift})

# --- Plot one trial ---
def plot_trial(block_number, trial_number):
    p_idx, tr_idx = block_number-1, trial_number-1
    x, t = as_1d(x_obj[p_idx, tr_idx]).astype(float), as_1d(t_obj[p_idx, tr_idx]).astype(float)
    fs, fe, used_center = fix_start[p_idx, tr_idx], fix_end[p_idx, tr_idx], cal_shift[p_idx, tr_idx]
    used_center = used_center if np.isfinite(used_center) else default_center
    pic_time = local_pic_time_ms(p_idx, tr_idx, t)
    t0, tz = t[0], t - t[0]
    pic_rel = pic_time - t0 if np.isfinite(pic_time) else np.nan
    rt_line = np.nan
    try: rt_line = float(rt_sec[tr_idx, p_idx]*1000.0) + pic_rel if np.isfinite(pic_rel) else np.nan
    except: pass

    used_lower, used_upper = used_center-100, used_center+100
    show_default = abs(used_center - default_center) > 1e-6

    plt.figure(figsize=(10, 5))
    plt.plot(tz, x, color='black', lw=1.0, label="X position")
    plt.axvline(0, color='green', ls='--', label="Cross onset")
    if np.isfinite(pic_rel): plt.axvline(pic_rel, color='red', ls='--', label="Picture onset")
    if np.isfinite(rt_line): plt.axvline(rt_line, color='purple', lw=1.5, label="Reaction time")
    if np.isfinite(fs) and np.isfinite(fe) and fe >= fs:
        in_fix = (tz >= fs) & (tz <= fe)
        plt.plot(tz[in_fix], x[in_fix], color='tab:red', lw=2.0, label="Dwell on cross")
    plt.axhline(used_lower, color='tab:blue', ls='-.', label="Lower threshold")
    plt.axhline(used_upper, color='tab:blue', ls='-.', label="Upper threshold")
    if show_default:
        plt.axhline(default_lower, color='orange', ls=':', lw=1.8, label="Original Lower Threshold")
        plt.axhline(default_upper, color='orange', ls=':', lw=1.8, label="Original Upper Threshold")
    plt.xlabel("Time (ms) relative to cross onset")
    plt.ylabel("X Position")
    plt.title(f"X Position (Participant {block_number}, Block {block_number}, Trial {trial_number})")
    plt.legend(loc="best")
    plt.tight_layout()
    plt.show()
