"""
Behavioral preprocessing — Cue/Arrow task

What this script does (brief):
  • Loads behavioral variables + epoched data (picture-locked, cross-locked, cross→cross)
  • Drops practice/test trials (0–4 and 49–53) across all structures
  • Marks incorrect trials as NaN (based on blockOrder, cue side, positivity/negativity, and pull/push rule)
  • Removes ultra-fast RTs (≤ 150 ms)
  • Computes per-block NaN counts and per-participant accuracy (valid-trial proportion)
  • Optionally removes specific blocks/participants (blocks_to_remove)
  • Plots RT distributions and winsorizes RTs (± 2 SD) for visualization
  • Saves all “_clean” .mat files (behavior, onsets, and epoched data), plus log(RT)
"""

import numpy as np
import pandas as pd
import scipy.io as sio
import matplotlib.pyplot as plt

# ===== Paths (as in your original) =====
ROOT = r"C:/Users/charl/OneDrive/Dokumente/Uni/7. Semester/Bachelor Thesis/ArrowTask"

# Behavior + onsets
MAT_BLOCK_ORDER = f"{ROOT}/blockOrder_legacy.mat"         # 'blockOrder'
MAT_CUE_SIDE    = f"{ROOT}/cueSide.mat"                   # 'cueSide'
MAT_DOM_EYE     = f"{ROOT}/dominantEye_Arrow.mat"         # 'dominantEye'
MAT_PIC_SEQ     = f"{ROOT}/pictureSequence.mat"           # 'pictureSequence'
MAT_RT          = f"{ROOT}/reactionTime.mat"              # 'reactionTime' (seconds)
MAT_REACT_TYPE  = f"{ROOT}/reactionType_legacy.mat"       # 'actualReaction'
MAT_PIC_IDX     = f"{ROOT}/pictureTimeIdx.mat"            # 'pictureTimeIdx'
MAT_CROSS_IDX   = f"{ROOT}/fixationTimeIdx.mat"           # 'crossTimeIdx'

# Cross→Cross (indices saved previously by your epoching)
MAT_PIC_ONSETS_C2C   = f"{ROOT}/pictureOnsets_idx_CrosstoCross.mat"  # 'pictureOnsetsIdx'
MAT_CROSS_ONSETS_C2C = f"{ROOT}/crossOnsets_idx_CrosstoCross.mat"    # 'crossOnsetsIdx'

# Epoched — Picture-locked
MAT_XR_PIC = f"{ROOT}/xPositionRight_epoched_PictureLocked.mat"   # 'xPositionRight'
MAT_XL_PIC = f"{ROOT}/xPositionLeft_epoched_PictureLocked.mat"    # 'xPositionLeft'
MAT_PR_PIC = f"{ROOT}/pupilSizeRight_epoched_PictureLocked.mat"   # 'pupilSizeRight'
MAT_PL_PIC = f"{ROOT}/pupilSizeLeft_epoched_PictureLocked.mat"    # 'pupilSizeLeft'
MAT_T_PIC  = f"{ROOT}/timeVector_epoched_PictureLocked.mat"       # 'timeVector'

# Epoched — Cross-locked (fixed length)
MAT_XR_CROSS = f"{ROOT}/xPositionRight_epoched_CrossLocked.mat"
MAT_XL_CROSS = f"{ROOT}/xPositionLeft_epoched_CrossLocked.mat"
MAT_PR_CROSS = f"{ROOT}/pupilSizeRight_epoched_CrossLocked.mat"
MAT_PL_CROSS = f"{ROOT}/pupilSizeLeft_epoched_CrossLocked.mat"
MAT_T_CROSS  = f"{ROOT}/timeVector_epoched_CrossLocked.mat"

# Epoched — Cross→Cross (variable length, MATLAB-cell style)
MAT_XR_FULL = f"{ROOT}/xPositionRight_epoched_CrosstoCross.mat"
MAT_XL_FULL = f"{ROOT}/xPositionLeft_epoched_CrosstoCross.mat"
MAT_PR_FULL = f"{ROOT}/pupilSizeRight_epoched_CrosstoCross.mat"
MAT_PL_FULL = f"{ROOT}/pupilSizeLeft_epoched_CrosstoCross.mat"
MAT_T_FULL  = f"{ROOT}/timeVector_epoched_CrosstoCross.mat"

# ===== Load behavior + onsets =====
blockOrder     = sio.loadmat(MAT_BLOCK_ORDER)["blockOrder"]              # (1, blocks), 'A'/'B'
cueSide        = sio.loadmat(MAT_CUE_SIDE)["cueSide"].astype(float)      # (trials, blocks) 0=left,1=right
dominantEye    = sio.loadmat(MAT_DOM_EYE)["dominantEye"]                 # (1, blocks) or similar
pictureSequence= sio.loadmat(MAT_PIC_SEQ)["pictureSequence"].astype(float)  # (trials, 2, blocks); <44=positive, ≥44=negative
reactionTimes  = sio.loadmat(MAT_RT)["reactionTime"].astype(float)       # (trials, blocks) in seconds
reactionType   = sio.loadmat(MAT_REACT_TYPE)["actualReaction"]           # (trials, blocks) 'pull'/'push'
pictureOnset   = sio.loadmat(MAT_PIC_IDX)["pictureTimeIdx"]              # (blocks,1) each cell length-98
crossOnset     = sio.loadmat(MAT_CROSS_IDX)["crossTimeIdx"]              # (blocks,1) each cell length-98

pic_onsets_c2c   = sio.loadmat(MAT_PIC_ONSETS_C2C)["pictureOnsetsIdx"]
cross_onsets_c2c = sio.loadmat(MAT_CROSS_ONSETS_C2C)["crossOnsetsIdx"]

# ===== Load epoched data =====
# Picture-locked
xPositionRightPic = sio.loadmat(MAT_XR_PIC)["xPositionRight"]
xPositionLeftPic  = sio.loadmat(MAT_XL_PIC)["xPositionLeft"]
PupilSizeRightPic = sio.loadmat(MAT_PR_PIC)["pupilSizeRight"]
PupilSizeLeftPic  = sio.loadmat(MAT_PL_PIC)["pupilSizeLeft"]
timeVectorPic     = sio.loadmat(MAT_T_PIC)["timeVector"]

# Cross-locked
xPositionRightCross = sio.loadmat(MAT_XR_CROSS)["xPositionRight"]
xPositionLeftCross  = sio.loadmat(MAT_XL_CROSS)["xPositionLeft"]
PupilSizeRightCross = sio.loadmat(MAT_PR_CROSS)["pupilSizeRight"]
PupilSizeLeftCross  = sio.loadmat(MAT_PL_CROSS)["pupilSizeLeft"]
timeVectorCross     = sio.loadmat(MAT_T_CROSS)["timeVector"]

# Cross→Cross
xPositionRightFull = sio.loadmat(MAT_XR_FULL)["xPositionRight"]
xPositionLeftFull  = sio.loadmat(MAT_XL_FULL)["xPositionLeft"]
PupilSizeRightFull = sio.loadmat(MAT_PR_FULL)["pupilSizeRight"]
PupilSizeLeftFull  = sio.loadmat(MAT_PL_FULL)["pupilSizeLeft"]
timeVectorFull     = sio.loadmat(MAT_T_FULL)["timeVector"]

# Cast onset arrays to float so we can assign NaN
for b in range(pictureOnset.shape[0]):
    pictureOnset[b, 0] = pictureOnset[b, 0].astype(float)
    crossOnset[b, 0]   = crossOnset[b, 0].astype(float)

# ===== Drop practice/test trials: indices [0..4] and [49..53] =====
to_drop = np.r_[0:5, 49:54]

def drop_trials_from_cell(cell_arr, drop_idx):
    """Remove columns (trials) from (blocks,1) MATLAB-cell-like arrays."""
    for i in range(cell_arr.shape[0]):
        arr = np.array(cell_arr[i, 0]).squeeze()
        cell_arr[i, 0] = np.delete(arr, drop_idx)
    return cell_arr

# picture-locked arrays: (blocks, trials, time)
xPositionRightPic = np.delete(xPositionRightPic, to_drop, axis=1)
xPositionLeftPic  = np.delete(xPositionLeftPic,  to_drop, axis=1)
PupilSizeRightPic = np.delete(PupilSizeRightPic, to_drop, axis=1)
PupilSizeLeftPic  = np.delete(PupilSizeLeftPic,  to_drop, axis=1)
timeVectorPic     = np.delete(timeVectorPic,     to_drop, axis=1)

# cross-locked arrays
xPositionRightCross = np.delete(xPositionRightCross, to_drop, axis=1)
xPositionLeftCross  = np.delete(xPositionLeftCross,  to_drop, axis=1)
PupilSizeRightCross = np.delete(PupilSizeRightCross, to_drop, axis=1)
PupilSizeLeftCross  = np.delete(PupilSizeLeftCross,  to_drop, axis=1)
timeVectorCross     = np.delete(timeVectorCross,     to_drop, axis=1)


# behavior arrays: (trials, blocks) or (trials, 2, blocks)
cueSide         = np.delete(cueSide,         to_drop, axis=0)
reactionTimes   = np.delete(reactionTimes,   to_drop, axis=0)
reactionType    = np.delete(reactionType,    to_drop, axis=0)
pictureSequence = np.delete(pictureSequence, to_drop, axis=0)

# onsets as cells
pictureOnset = drop_trials_from_cell(pictureOnset, to_drop)
crossOnset   = drop_trials_from_cell(crossOnset,   to_drop)

num_trials = cueSide.shape[0]   
num_blocks = cueSide.shape[1]

# ===== Small helper: invalidate a single trial everywhere =====
def invalidate_trial(block, trial):
    # behavior
    cueSide[trial, block] = np.nan
    pictureSequence[trial, :, block] = np.nan
    reactionTimes[trial, block] = np.nan
    reactionType[trial, block] = np.nan

    # picture-locked
    xPositionRightPic[block, trial, :] = np.nan
    xPositionLeftPic[block,  trial, :] = np.nan
    PupilSizeRightPic[block,  trial, :] = np.nan
    PupilSizeLeftPic[block,   trial, :] = np.nan
    timeVectorPic[block,      trial, :] = np.nan

    # cross-locked
    xPositionRightCross[block, trial, :] = np.nan
    xPositionLeftCross[block,  trial, :] = np.nan
    PupilSizeRightCross[block, trial, :] = np.nan
    PupilSizeLeftCross[block,  trial, :] = np.nan
    timeVectorCross[block,     trial, :] = np.nan

    # cross→cross (object arrays: (blocks, trials))
    xPositionRightFull[block, trial] = np.nan
    xPositionLeftFull[block,  trial] = np.nan
    PupilSizeRightFull[block, trial] = np.nan
    PupilSizeLeftFull[block,  trial] = np.nan
    timeVectorFull[block,     trial] = np.nan

    # onsets (cells)
    if trial < len(pictureOnset[block, 0]):
        pictureOnset[block, 0][trial] = np.nan
    if trial < len(crossOnset[block, 0]):
        crossOnset[block, 0][trial] = np.nan


# ===== Mark incorrect trials as NaN (rule-based) =====
correct = 0
incorrect = 0

def is_positive(pic_id):  # <44 → positive, ≥44 → negative
    return pic_id < 44

for block in range(num_blocks):
    order = str(blockOrder[0, block])  # 'A' or 'B'
    # mapping: 'A' → congruent first half (0..43), incongruent second half (44..87)
    #          'B' → incongruent first, congruent second
    for trial in range(num_trials):
        left_id  = pictureSequence[trial, 0, block]
        right_id = pictureSequence[trial, 1, block]
        side     = cueSide[trial, block]               # 0=left, 1=right
        resp     = str(reactionType[trial, block])     # 'pull' or 'push'

        if np.isnan(side) or np.isnan(left_id) or np.isnan(right_id) or pd.isna(resp):
            continue

        in_first_half = (trial < 44)
        if order == "A":
            congruent = in_first_half
        elif order == "B":
            congruent = not in_first_half

        # pick the cued picture id (valence depends on side)
        pic_id = left_id if side == 0 else right_id
        pos    = is_positive(pic_id)

        # correct response under congruent/incongruent mapping
        # congruent:  positive→pull, negative→push
        # incongruent:positive→push, negative→pull
        if congruent:
            should_pull = pos
        else:
            should_pull = not pos

        is_pull = (resp == "pull")
        is_correct = (is_pull == should_pull)

        if is_correct:
            correct += 1
        else:
            incorrect += 1
            invalidate_trial(block, trial)

print(f"number of correct trials: {correct}")
print(f"number of incorrect trials: {incorrect}")

# ===== Remove too fast trials (RT ≤ 150 ms) =====
fast_trials = np.where(reactionTimes <= 0.150)  # (trial_idx, block_idx)
for trial, block in zip(*fast_trials):
    invalidate_trial(block, trial)
print(f"Removed {len(fast_trials[0])} trials with RT ≤ 150 ms")

# ===== Summaries =====
def count_nan_trials_3d(name, arr):
    if arr.ndim != 3:
        print(f"{name}: unexpected shape {arr.shape}")
        return
    full_nan = np.sum(np.all(np.isnan(arr), axis=2))
    print(f"{name}: {full_nan} full trials set to NaN (sum over blocks*trials)")

print("\n=== NaN Trials Summary (full-NaN epochs) ===")
for nm, arr in [
    ("xPositionRightPic", xPositionRightPic),
    ("xPositionLeftPic",  xPositionLeftPic),
    ("PupilSizeRightPic", PupilSizeRightPic),
    ("PupilSizeLeftPic",  PupilSizeLeftPic),
    ("timeVectorPic",     timeVectorPic),
    ("xPositionRightCross", xPositionRightCross),
    ("xPositionLeftCross",  xPositionLeftCross),
    ("PupilSizeRightCross", PupilSizeRightCross),
    ("PupilSizeLeftCross",  PupilSizeLeftCross),
    ("timeVectorCross",     timeVectorCross),
]:
    count_nan_trials_3d(nm, arr)

# Per-block NaN count in key behavioral vars
nan_trials_per_block = np.zeros(num_blocks, dtype=int)
for b in range(num_blocks):
    for t in range(num_trials):
        if (np.isnan(cueSide[t, b]) or
            np.isnan(reactionTimes[t, b]) or
            pd.isna(reactionType[t, b]) or
            np.isnan(pictureSequence[t, :, b]).any()):
            nan_trials_per_block[b] += 1
for b, cnt in enumerate(nan_trials_per_block):
    print(f"Block {b}: {cnt} NaN trials")

# Per-participant accuracy 
blocks_per_participant = 2
num_participants = num_blocks // blocks_per_participant
trials_per_block = num_trials
for p in range(num_participants):
    b1, b2 = p * 2, p * 2 + 1
    valid = 0
    for b in [b1, b2]:
        for t in range(trials_per_block):
            if (not np.isnan(cueSide[t, b]) and
                not np.isnan(reactionTimes[t, b]) and
                not pd.isna(reactionType[t, b]) and
                not np.isnan(pictureSequence[t, :, b]).any()):
                valid += 1
    acc = valid / (trials_per_block * 2)
    print(f"Participant {p}: {acc:.2%} accuracy")

# ===== Remove participants to be removed (excluded or low accuracy) =====
blocks_to_remove = [34, 35, 48, 49, 50, 51]  # keep your original choice
def remove_blocks(arr, axis=1):
    if isinstance(arr, np.ndarray) and arr.ndim >= axis + 1:
        return np.delete(arr, blocks_to_remove, axis=axis)
    return arr


dominantEye    = np.delete(dominantEye, blocks_to_remove, axis=1)
cueSide        = remove_blocks(cueSide,        axis=1)
reactionTimes  = remove_blocks(reactionTimes,  axis=1)
reactionType   = remove_blocks(reactionType,   axis=1)
pictureSequence= remove_blocks(pictureSequence,axis=2)
blockOrder     = np.delete(blockOrder, blocks_to_remove, axis=1)
pictureOnset   = np.delete(pictureOnset, blocks_to_remove, axis=0)
crossOnset     = np.delete(crossOnset,   blocks_to_remove, axis=0)


xPositionRightPic = remove_blocks(xPositionRightPic, axis=0)
xPositionLeftPic  = remove_blocks(xPositionLeftPic,  axis=0)
PupilSizeRightPic = remove_blocks(PupilSizeRightPic, axis=0)
PupilSizeLeftPic  = remove_blocks(PupilSizeLeftPic,  axis=0)
timeVectorPic     = remove_blocks(timeVectorPic,     axis=0)

xPositionRightCross = remove_blocks(xPositionRightCross, axis=0)
xPositionLeftCross  = remove_blocks(xPositionLeftCross,  axis=0)
PupilSizeRightCross = remove_blocks(PupilSizeRightCross, axis=0)
PupilSizeLeftCross  = remove_blocks(PupilSizeLeftCross,  axis=0)
timeVectorCross     = remove_blocks(timeVectorCross,     axis=0)

xPositionRightFull  = remove_blocks(xPositionRightFull,  axis=0)
xPositionLeftFull   = remove_blocks(xPositionLeftFull,   axis=0)
PupilSizeRightFull  = remove_blocks(PupilSizeRightFull,  axis=0)
PupilSizeLeftFull   = remove_blocks(PupilSizeLeftFull,   axis=0)
timeVectorFull      = remove_blocks(timeVectorFull,      axis=0)

# ===== RT distributions + winsorizing (for plots/report) =====
rt_clean = reactionTimes[~np.isnan(reactionTimes)]
plt.figure(figsize=(10, 6))
plt.hist(rt_clean, bins=50, edgecolor='black')
plt.title("Distribution of Reaction Times")
plt.xlabel("Reaction Time (s)")
plt.ylabel("Frequency")
plt.grid(True); plt.tight_layout(); plt.show()

mean_rt = np.mean(rt_clean); std_rt = np.std(rt_clean)
lb = mean_rt - 2 * std_rt
ub = mean_rt + 2 * std_rt
rt_wins = np.clip(rt_clean, lb, ub)

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1); plt.hist(rt_clean, bins=50, edgecolor='black'); plt.title("Original RTs"); plt.xlabel("RT (s)"); plt.grid(True)
plt.subplot(1, 2, 2); plt.hist(rt_wins, bins=50, edgecolor='black'); plt.title("Winsorized RTs (±2 SD)"); plt.xlabel("RT (s)"); plt.grid(True)
plt.tight_layout(); plt.show()

# Log-transform RTs 
reactionTimes_log = np.copy(reactionTimes)
mask = ~np.isnan(reactionTimes) & (reactionTimes > 0)
reactionTimes_log[mask] = np.log(reactionTimes[mask])

# ===== Save all cleaned outputs =====
savemat = sio.savemat
savemat("cueSide_clean.mat",               {"cueSide": cueSide})
savemat("dominantEye_clean.mat",           {"dominantEye": dominantEye})
savemat("pictureSequence_clean.mat",       {"pictureSequence": pictureSequence})
savemat("reactionTimes_clean.mat",         {"reactionTimes": reactionTimes})
savemat("reactionType_clean.mat",          {"reactionType": reactionType})
savemat("blockOrder_clean.mat",            {"blockOrder": blockOrder})
savemat("pictureOnsetIdx_clean.mat",       {"pictureOnsetIdx": pictureOnset})
savemat("fixationOnsetIdx_clean.mat",      {"fixationOnsetIdx": crossOnset})
savemat("crossOnsets_idx_CrosstoCross.mat",   {"crossOnsetsIdx": cross_onsets_c2c})
savemat("pictureOnsets_idx_CrosstoCross.mat", {"pictureOnsetsIdx": pic_onsets_c2c})

savemat("xPositionRightPic_clean.mat", {"xPositionRight": xPositionRightPic})
savemat("xPositionLeftPic_clean.mat",  {"xPositionLeft":  xPositionLeftPic})
savemat("PupilSizeRightPic_clean.mat", {"pupilSizeRight": PupilSizeRightPic})
savemat("PupilSizeLeftPic_clean.mat",  {"pupilSizeLeft":  PupilSizeLeftPic})
savemat("timeVectorPic_clean.mat",     {"timeVector":     timeVectorPic})

savemat("xPositionRightCross_clean.mat", {"xPositionRight": xPositionRightCross})
savemat("xPositionLeftCross_clean.mat",  {"xPositionLeft":  xPositionLeftCross})
savemat("PupilSizeRightCross_clean.mat", {"pupilSizeRight": PupilSizeRightCross})
savemat("PupilSizeLeftCross_clean.mat",  {"pupilSizeLeft":  PupilSizeLeftCross})
savemat("timeVectorCross_clean.mat",     {"timeVector":     timeVectorCross})

savemat("xPositionRightCrosstoCross_clean.mat", {"xPositionRight": xPositionRightFull})
savemat("xPositionLeftCrosstoCross_clean.mat",  {"xPositionLeft":  xPositionLeftFull})
savemat("PupilSizeRightCrosstoCross_clean.mat", {"pupilSizeRight": PupilSizeRightFull})
savemat("PupilSizeLeftCrosstoCross_clean.mat",  {"pupilSizeLeft":  PupilSizeLeftFull})
savemat("timeVectorCrosstoCross_clean.mat",     {"timeVector":     timeVectorFull})

savemat("reactionTimes_log.mat", {"reactionTimes": reactionTimes_log})

print("✅ Saved all *_clean.mat and reactionTimes_log.mat")
