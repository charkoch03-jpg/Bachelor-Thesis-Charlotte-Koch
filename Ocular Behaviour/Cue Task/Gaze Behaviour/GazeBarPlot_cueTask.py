import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt

# --- Paths ---
ROOT = r"C:/Users/charl/OneDrive/Dokumente/Uni/7. Semester/Bachelor Thesis/ArrowTask"
MAT  = lambda name: f"{ROOT}/{name}"

# --- Load ---
first_side = sio.loadmat(MAT("firstFixationSide.mat"))["firstFixationSide"].astype(float)
last_side  = sio.loadmat(MAT("lastFixationSide.mat"))["lastFixationSide"].astype(float)

first_st = sio.loadmat(MAT("firstFixationStartTimes.mat"))["firstFixationStartTimes"].astype(float)
first_en = sio.loadmat(MAT("firstFixationEndTimes.mat"))["firstFixationEndTimes"].astype(float)
last_st  = sio.loadmat(MAT("lastFixationStartTimes.mat"))["lastFixationStartTimes"].astype(float)
last_en  = sio.loadmat(MAT("lastFixationEndTimes.mat"))["lastFixationEndTimes"].astype(float)

first_dur = first_en - first_st
last_dur  = last_en - last_st

# --- Same vs different side ---
same_side = 0
different_side = 0
P, T = first_side.shape

for p in range(P):
    for t in range(T):
        a, b = first_side[p, t], last_side[p, t]
        if np.isnan(a) or np.isnan(b): 
            continue
        if a == b: same_side += 1
        else:      different_side += 1

print("same side", same_side)
print("different side", different_side)

# --- Plot #1 ---
labels = ["Same side", "Different side"]
counts = [same_side, different_side]

fig, ax = plt.subplots(figsize=(5, 3.5))
bars = ax.bar(labels, counts, edgecolor="black", linewidth=0.8)
ax.set_ylabel("Number of trials")
ax.set_title("Side of first vs. last picture dwell (Cue task)")
ax.grid(axis="y", alpha=0.35)
for b in bars:
    h = b.get_height()
    ax.text(b.get_x()+b.get_width()/2, h, f"{int(h)}", ha="center", va="bottom", fontsize=9)
plt.tight_layout()
plt.show()

# --- 3 categories ---
only_first = 0
back_to_first = 0

for p in range(P):
    for t in range(T):
        a, b = first_side[p, t], last_side[p, t]
        if np.isnan(a) or np.isnan(b): 
            continue
        if a == b:
            fd, ld = first_dur[p, t], last_dur[p, t]
            if np.isfinite(fd) and np.isfinite(ld) and np.isclose(fd, ld, atol=1e-6):
                only_first += 1
            else:
                back_to_first += 1

labels = ["First only", "First → Second", "First → Second → First"]
counts = [only_first, different_side, back_to_first]

fig, ax = plt.subplots(figsize=(6, 4))
bars = ax.bar(range(len(labels)), counts, edgecolor="black", linewidth=0.8)
ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels)
ax.set_ylabel("Number of trials")
ax.set_title("Gaze sequence counts (Cue task)")
ax.grid(axis="y", alpha=0.35)
for b, c in zip(bars, counts):
    ax.text(b.get_x()+b.get_width()/2, b.get_height(), f"{int(c)}", ha="center", va="bottom", fontsize=9)
plt.tight_layout()
plt.show()
