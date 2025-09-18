import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# --- Load ---
xR = sio.loadmat(r"C:/Users/charl/OneDrive/Dokumente/Uni/7. Semester/Bachelor Thesis/Eye Data Experiment1Task2/Experiment1_Task2/xPositionRight.mat")["xPositionRight"]
xL = sio.loadmat(r"C:/Users/charl/OneDrive/Dokumente/Uni/7. Semester/Bachelor Thesis/Eye Data Experiment1Task2/Experiment1_Task2/xPositionLeft.mat")["xPositionLeft"]
t3 = sio.loadmat( r"C:/Users/charl/OneDrive/Dokumente/Uni/7. Semester/Bachelor Thesis/Eye Data Experiment1Task2/Experiment1_Task2/timeVector.mat")["timeVector"]
dom = pd.read_csv( r"C:/Users/charl/OneDrive/Dokumente/Uni/7. Semester/Bachelor Thesis/Eye Data Experiment1Task2/Experiment1_Task2/dominantEye.csv", header=None).values.flatten()
cal = sio.loadmat( r"C:/Users/charl/OneDrive/Dokumente/Uni/7. Semester/Bachelor Thesis/Eye Data Experiment1Task2/calibrationShifts.mat")["calibrationShifts"]

# --- Remove bad participants + test trials ---
bad = [42, 43, 120, 121]
xL = np.delete(xL, bad, axis=0); xR = np.delete(xR, bad, axis=0)
t3 = np.delete(t3, bad, axis=0); dom = np.delete(dom, bad, axis=0); cal = np.delete(cal, bad, axis=0)
xL = xL[:, 4:, :]; xR = xR[:, 4:, :]; t3 = t3[:, 4:, :]; cal = cal[:, 4:]

# --- Parameterss ---
idx_on = 500
thr = 100
P, T, S = xR.shape
NTP = 1500                     # 500 before + 1000 after pic onset
time_rel = np.arange(-500, 1000)

# --- Counters ---
first_pic = np.full((P, T), np.nan)   
cross = np.zeros(NTP)
first = np.zeros(NTP)
second = np.zeros(NTP)
back_to_first = np.zeros(NTP)

# --- Trial loop ---
for p in range(P):
    for tr in range(T):
        x = xR[p, tr] if dom[p] == 1 else xL[p, tr]
        c = cal[p, tr]
        # first looked side after ~onset
        for t in range(idx_on, NTP):
            if x[t] > c + thr: first_pic[p, tr] = 1; break
            if x[t] < c - thr: first_pic[p, tr] = 0; break

        seen_second = False
        for t in range(NTP):
            if (c - thr) < x[t] < (c + thr):
                cross[t] += 1
            elif x[t] < (c - thr):
                if first_pic[p, tr] == 1:
                    second[t] += 1;  seen_second |= (t >= idx_on)
                else:
                    if t >= idx_on and seen_second: back_to_first[t] += 1
                    else:                            first[t] += 1
            elif x[t] > (c + thr):
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
plt.title('Gaze Behavior over Time (Free Viewing)')
plt.legend(loc='upper right')
plt.tight_layout()
plt.show()
