import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt

# --- Load ---
ROOT = r"C:/Users/charl/OneDrive/Dokumente/Uni/7. Semester/Bachelor Thesis/ArrowTask"
xR = sio.loadmat(f"{ROOT}/xPositionRightPic_clean.mat")["xPositionRight"]
xL = sio.loadmat(f"{ROOT}/xPositionLeftPic_clean.mat")["xPositionLeft"]
t3 = sio.loadmat( f"{ROOT}/timeVectorPic_clean.mat")["timeVector"]
dom = sio.loadmat(f"{ROOT}/dominantEye_clean.mat")["dominantEye"]  
# --- Select eye per participant ---
P, T, S = t3.shape
x = np.empty_like(xR)
for p in range(P):
    x[p, :] = xR[p, :] if int(dom[0, p]) == 0 else xL[p, :]

# --- Params ---
idx_on = 500
center = 960
thr = 100
NTP = S
time_rel = np.arange(-idx_on, NTP - idx_on)

# --- Counters ---
first_pic = np.full((P, T), np.nan)   # 0=left, 1=right
cross = np.zeros(NTP)
first = np.zeros(NTP)
second = np.zeros(NTP)
back_to_first = np.zeros(NTP)

# --- Trial loop ---
for p in range(P):
    for tr in range(T):
        xpos = x[p, tr]

        # first looked side
        for t in range(idx_on + 40, NTP):
            if xpos[t] > center + thr: first_pic[p, tr] = 1; break
            if xpos[t] < center - thr: first_pic[p, tr] = 0; break

        seen_second = False
        for t in range(NTP):
            if (center - thr) < xpos[t] < (center + thr):
                cross[t] += 1
            elif xpos[t] < (center - thr):
                if first_pic[p, tr] == 1:
                    second[t] += 1;  seen_second |= (t >= idx_on)
                else:
                    if t >= idx_on and seen_second: back_to_first[t] += 1
                    else:                            first[t] += 1
            elif xpos[t] > (center + thr):
                if first_pic[p, tr] == 0:
                    second[t] += 1;  seen_second |= (t >= idx_on)
                else:
                    if t >= idx_on and seen_second: back_to_first[t] += 1
                    else:                            first[t] += 1

# --- Percent of trials ---
total_trials = P * T
cross_p  = (cross  / total_trials) * 100
first_p  = (first  / total_trials) * 100
second_p = (second / total_trials) * 100
back_p   = (back_to_first / total_trials) * 100

# --- Plot ---
plt.figure(figsize=(12, 6))
plt.stackplot(time_rel * 2, cross_p, first_p, second_p, back_p,
              labels=["Cross (Center)", "First Picture", "Second Picture", "Back to First"],
              colors=['gray', 'blue', 'orange', 'purple'])
plt.axvline(0, color='k', linestyle='--', label='Picture Onset')
plt.xlabel('Time in ms (relative to picture onset)')
plt.ylabel('Percentage of Trials')
plt.title('Gaze Behavior Over Time (Cue Task)')
plt.legend(loc='upper right')
plt.tight_layout()
plt.show()
