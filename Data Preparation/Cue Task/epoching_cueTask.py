"""
Epoching — Arrow/Cue task
Modes:
  1) Cross→Cross (variable length; final epoch until last picture + 2000 ms)
  2) Picture-locked (fixed pre/post samples)
  3) RT-locked (−1000..+1000 ms around each trial's RT)
"""

import numpy as np
import scipy.io as sio

# ===== Config =====
FS_HZ = 500.0
EXTRA_LAST_MS = 2000  # extend final Cross→Cross epoch to (last picture + 2000 ms), capped at array end

# Picture-locked window
PRE_PIC_SAMPLES  = 500   # ~1000 ms at 500 Hz
POST_PIC_SAMPLES = 720   # ~1440 ms at 500 Hz
PIC_EPOCH_LEN    = PRE_PIC_SAMPLES + POST_PIC_SAMPLES

# RT-locked window
PRE_RT_MS  = 1000
POST_RT_MS = 1000
N_PRE  = int(round(PRE_RT_MS  * FS_HZ / 1000.0))   
N_POST = int(round(POST_RT_MS * FS_HZ / 1000.0))   
RT_EPOCH_LEN = N_PRE + N_POST
MS_PER_SAMPLE = 1000.0 / FS_HZ
TIME_MS = (np.arange(RT_EPOCH_LEN) - N_PRE) * MS_PER_SAMPLE  # shared time axis (ms)

# ===== Paths =====
ROOT = r"C:/Users/charl/OneDrive/Dokumente/Uni/7. Semester/Bachelor Thesis/ArrowTask"

MAT_PIC_IDX   = f"{ROOT}/pictureTimeIdx.mat"          
MAT_CROSS_IDX = f"{ROOT}/fixationTimeIdx.mat"         
MAT_PR        = f"{ROOT}/pupilSizeRight_cleanedInterpolated.mat"  
MAT_PL        = f"{ROOT}/pupilSizeLeft_cleanedInterpolated.mat"   
MAT_XR        = f"{ROOT}/xPositionRightContinuos.mat"             
MAT_XL        = f"{ROOT}/xPositionLeftContinuos.mat"             
MAT_T         = f"{ROOT}/timeVectorContinuos.mat"                 
MAT_RT        = f"{ROOT}/reactionTimes_clean.mat"                

# Outputs — Cross→Cross
OUT_C2C_L    = "pupilSizeLeft_epoched_CrosstoCross.mat"
OUT_C2C_R    = "pupilSizeRight_epoched_CrosstoCross.mat"
OUT_C2C_XL   = "xPositionLeft_epoched_CrosstoCross.mat"
OUT_C2C_XR   = "xPositionRight_epoched_CrosstoCross.mat"
OUT_C2C_T    = "timeVector_epoched_CrosstoCross.mat"
OUT_C2C_CROSS = "crossOnsets_idx_CrosstoCross.mat"
OUT_C2C_PIC   = "pictureOnsets_idx_CrosstoCross.mat"

# Outputs — Picture-locked
OUT_PIC_L  = "pupilSizeLeft_epoched_PictureLocked.mat"
OUT_PIC_R  = "pupilSizeRight_epoched_PictureLocked.mat"
OUT_PIC_XL = "xPositionLeft_epoched_PictureLocked.mat"
OUT_PIC_XR = "xPositionRight_epoched_PictureLocked.mat"
OUT_PIC_T  = "timeVector_epoched_PictureLocked.mat"

# Outputs — RT-locked
OUT_RT_L   = "pupilSizeLeft_epoched_RTLocked.mat"
OUT_RT_R   = "pupilSizeRight_epoched_RTLocked.mat"
OUT_RT_XL  = "xPositionLeft_epoched_RTLocked.mat"
OUT_RT_XR  = "xPositionRight_epoched_RTLocked.mat"
OUT_RT_IDX = "rt_abs_idx_RTLocked.mat"

# ===== Helpers =====
def _obj_to_sorted_1d(arr_like):
    """MATLAB cell/obj → sorted 1D int array (drop NaNs)."""
    try:
        if isinstance(arr_like, np.ndarray) and arr_like.dtype == "O":
            arr_like = arr_like.item()
    except Exception:
        pass
    arr = np.array(arr_like).squeeze()
    if arr.size == 0:
        return np.array([], dtype=int)
    arr = np.asarray(arr, dtype=float)
    arr = arr[~np.isnan(arr)]
    return np.sort(arr.astype(int))

def _list_to_matcell(block_list):
    """Python nested [blocks][trials](arrays) → MATLAB cell-like object array."""
    return np.array([[np.array(trial) for trial in block] for block in block_list], dtype=object)

# ===== Load data =====
picOnsetIdx   = sio.loadmat(MAT_PIC_IDX)["pictureTimeIdx"]
_cross_obj    = sio.loadmat(MAT_CROSS_IDX)
cross_key     = "crossTimeIdx" if "crossTimeIdx" in _cross_obj else "fixationTimeIdx"
crossOnsetIdx = _cross_obj[cross_key]

PupilSizeRight = sio.loadmat(MAT_PR)["pupilSizeRight"]
PupilSizeLeft  = sio.loadmat(MAT_PL)["pupilSizeLeft"]
xPositionRight = sio.loadmat(MAT_XR)["xPositionRight"]
xPositionLeft  = sio.loadmat(MAT_XL)["xPositionLeft"]
timeVector     = sio.loadmat(MAT_T)["timeVector"]
reactionTimes  = sio.loadmat(MAT_RT)["reactionTimes"]  # in seconds

num_blocks = PupilSizeLeft.shape[0]

# Convenience per-block arrays
pL_blocks = [PupilSizeLeft[b, 0].squeeze()  for b in range(num_blocks)]
pR_blocks = [PupilSizeRight[b, 0].squeeze() for b in range(num_blocks)]
xL_blocks = [xPositionLeft[b, 0].squeeze()  for b in range(num_blocks)]
xR_blocks = [xPositionRight[b, 0].squeeze() for b in range(num_blocks)]
t_blocks  = [timeVector[b, 0].squeeze()     for b in range(num_blocks)]

# Onset indices (sorted 1D per block)
pic_idxs_blocks   = [_obj_to_sorted_1d(picOnsetIdx[b, 0])   for b in range(num_blocks)]
cross_idxs_blocks = [_obj_to_sorted_1d(crossOnsetIdx[b, 0]) for b in range(num_blocks)]

# ==============================
# 1) Cross→Cross (variable length)
# ==============================
epoched_pupilLeft  = []
epoched_pupilRight = []
epoched_xLeft      = []
epoched_xRight     = []
epoched_time       = []
cross_onsets_by_trial = []
pic_onsets_by_trial   = []

for block in range(num_blocks):
    pupilL = pL_blocks[block]; pupilR = pR_blocks[block]
    xL     = xL_blocks[block]; xR     = xR_blocks[block]
    tVec   = t_blocks[block]
    cross_idxs = cross_idxs_blocks[block]
    pic_idxs   = pic_idxs_blocks[block]

    block_epochs_L, block_epochs_R = [], []
    block_xL, block_xR, block_times = [], [], []
    block_cross_onsets, block_pic_onsets = [], []

    # cross → next cross
    for t in range(len(cross_idxs) - 1):
        start = int(cross_idxs[t])
        end   = int(cross_idxs[t + 1])

        block_epochs_L.append(pupilL[start:end])
        block_epochs_R.append(pupilR[start:end])
        block_xL.append(xL[start:end])
        block_xR.append(xR[start:end])
        block_times.append(tVec[start:end])
        block_cross_onsets.append(start)

        in_win = pic_idxs[(pic_idxs >= start) & (pic_idxs < end)]
        block_pic_onsets.append(int(in_win[0]) if in_win.size else np.nan)

    # tail: last cross → (last picture + EXTRA_LAST_MS), capped
    if len(cross_idxs) > 0 and pic_idxs.size > 0:
        last_cross = int(cross_idxs[-1])
        last_pic   = int(pic_idxs[-1])
        tail_end   = min(last_pic + int(round(EXTRA_LAST_MS * FS_HZ / 1000.0)), pupilL.size)
        if tail_end > last_cross:
            block_epochs_L.append(pupilL[last_cross:tail_end])
            block_epochs_R.append(pupilR[last_cross:tail_end])
            block_xL.append(xL[last_cross:tail_end])
            block_xR.append(xR[last_cross:tail_end])
            block_times.append(tVec[last_cross:tail_end])
            block_cross_onsets.append(last_cross)

            in_tail = pic_idxs[(pic_idxs >= last_cross) & (pic_idxs < tail_end)]
            block_pic_onsets.append(int(in_tail[0]) if in_tail.size else np.nan)

    epoched_pupilLeft.append(block_epochs_L)
    epoched_pupilRight.append(block_epochs_R)
    epoched_xLeft.append(block_xL)
    epoched_xRight.append(block_xR)
    epoched_time.append(block_times)
    cross_onsets_by_trial.append(block_cross_onsets)
    pic_onsets_by_trial.append(block_pic_onsets)

# save Cross→Cross (cell-style)
sio.savemat(OUT_C2C_L,    {"pupilSizeLeft":  _list_to_matcell(epoched_pupilLeft)})
sio.savemat(OUT_C2C_R,    {"pupilSizeRight": _list_to_matcell(epoched_pupilRight)})
sio.savemat(OUT_C2C_XL,   {"xPositionLeft":  _list_to_matcell(epoched_xLeft)})
sio.savemat(OUT_C2C_XR,   {"xPositionRight": _list_to_matcell(epoched_xRight)})
sio.savemat(OUT_C2C_T,    {"timeVector":     _list_to_matcell(epoched_time)})
sio.savemat(OUT_C2C_CROSS, {"crossOnsetsIdx":   _list_to_matcell(cross_onsets_by_trial)})
sio.savemat(OUT_C2C_PIC,   {"pictureOnsetsIdx": _list_to_matcell(pic_onsets_by_trial)})

# ==============================
# 2) Picture-locked (fixed pre/post)
# ==============================
max_trials = max((len(v) for v in pic_idxs_blocks), default=0)
if max_trials > 0:
    pupil_left_epochs  = np.full((num_blocks, max_trials, PIC_EPOCH_LEN), np.nan)
    pupil_right_epochs = np.full((num_blocks, max_trials, PIC_EPOCH_LEN), np.nan)
    x_left_epochs      = np.full((num_blocks, max_trials, PIC_EPOCH_LEN), np.nan)
    x_right_epochs     = np.full((num_blocks, max_trials, PIC_EPOCH_LEN), np.nan)
    time_epochs        = np.full((num_blocks, max_trials, PIC_EPOCH_LEN), np.nan)

    for block in range(num_blocks):
        pupilL = pL_blocks[block]; pupilR = pR_blocks[block]
        xL     = xL_blocks[block]; xR     = xR_blocks[block]
        tVec   = t_blocks[block]
        pic_idxs = pic_idxs_blocks[block]

        for t, onset in enumerate(pic_idxs):
            if t >= max_trials:
                break
            start = int(onset) - PRE_PIC_SAMPLES
            end   = int(onset) + POST_PIC_SAMPLES
            if start < 0 or end > pupilL.size or end <= start:
                continue

            pupil_left_epochs[block, t, :]  = pupilL[start:end]
            pupil_right_epochs[block, t, :] = pupilR[start:end]
            x_left_epochs[block, t, :]      = xL[start:end]
            x_right_epochs[block, t, :]     = xR[start:end]
            time_epochs[block, t, :]        = tVec[start:end]

    sio.savemat(OUT_PIC_L,  {"pupilSizeLeft":  pupil_left_epochs})
    sio.savemat(OUT_PIC_R,  {"pupilSizeRight": pupil_right_epochs})
    sio.savemat(OUT_PIC_XL, {"xPositionLeft":  x_left_epochs})
    sio.savemat(OUT_PIC_XR, {"xPositionRight": x_right_epochs})
    sio.savemat(OUT_PIC_T,  {"timeVector":     time_epochs})

# ==============================
# 3) RT-locked (−1000..+1000 ms)
# ==============================
max_trials_rt = max_trials
rt_pupilLeft  = np.full((num_blocks, max_trials_rt, RT_EPOCH_LEN), np.nan)
rt_pupilRight = np.full((num_blocks, max_trials_rt, RT_EPOCH_LEN), np.nan)
rt_xLeft      = np.full((num_blocks, max_trials_rt, RT_EPOCH_LEN), np.nan)
rt_xRight     = np.full((num_blocks, max_trials_rt, RT_EPOCH_LEN), np.nan)
rt_abs_idx    = np.full((num_blocks, max_trials_rt), np.nan)

for block in range(num_blocks):
    pupilL = pL_blocks[block]; pupilR = pR_blocks[block]
    xL     = xL_blocks[block]; xR     = xR_blocks[block]
    pic_idxs = pic_idxs_blocks[block]

    # pick the axis that matches trials
    if reactionTimes.ndim == 2 and reactionTimes.shape[0] == num_blocks:
        rts_sec = reactionTimes[block, :].squeeze()
    elif reactionTimes.ndim == 2 and reactionTimes.shape[1] == num_blocks:
        rts_sec = reactionTimes[:, block].squeeze()
    else:
        rts_sec = reactionTimes.squeeze()

    for t, onset_idx in enumerate(pic_idxs):
        if t >= max_trials_rt:
            break
        try:
            rt_sec = float(rts_sec[t])  # seconds relative to picture onset
        except Exception:
            continue
        if not np.isfinite(rt_sec):
            continue

        rt_idx = int(onset_idx) + int(round(rt_sec * FS_HZ))
        start  = rt_idx - N_PRE
        end    = rt_idx + N_POST
        if start < 0 or end > pupilL.size or end <= start:
            continue

        rt_pupilLeft[block, t, :]  = pupilL[start:end]
        rt_pupilRight[block, t, :] = pupilR[start:end]
        rt_xLeft[block, t, :]      = xL[start:end]
        rt_xRight[block, t, :]     = xR[start:end]
        rt_abs_idx[block, t]       = rt_idx

# save RT-locked (+ shared time axis)
sio.savemat(OUT_RT_L,  {"pupilSizeLeft":  rt_pupilLeft,  "time_ms": TIME_MS})
sio.savemat(OUT_RT_R,  {"pupilSizeRight": rt_pupilRight, "time_ms": TIME_MS})
sio.savemat(OUT_RT_XL, {"xPositionLeft":  rt_xLeft,      "time_ms": TIME_MS})
sio.savemat(OUT_RT_XR, {"xPositionRight": rt_xRight,     "time_ms": TIME_MS})
sio.savemat(OUT_RT_IDX, {"rtAbsIdx": rt_abs_idx})

print("✅ Saved: Cross→Cross, Picture-locked, and RT-locked epochs.")


