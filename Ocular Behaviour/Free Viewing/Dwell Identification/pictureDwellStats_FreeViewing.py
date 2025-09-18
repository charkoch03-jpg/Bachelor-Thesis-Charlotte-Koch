import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt

# ---------- LOAD ----------
ROOT = r"C:/Users/charl/OneDrive/Dokumente/Uni/7. Semester/Bachelor Thesis/Eye Data Experiment1Task2/Experiment1_Task2"
first_st = sio.loadmat(f"{ROOT}/fixationPictureStartTimes.mat")["fixationStartTimes"]
first_en = sio.loadmat(f"{ROOT}/fixationPictureEndTimes.mat")  ["fixationEndTimes"]
last_st  = sio.loadmat(f"{ROOT}/lastFixationPictureStartTimes.mat")["lastFixationPictureStartTimes"]
last_en  = sio.loadmat(f"{ROOT}/lastFixationPictureEndTimes.mat")  ["lastFixationPictureEndTimes"]
tvec     = sio.loadmat(f"{ROOT}/timeVector.mat")["timeVector"]  # (P,T,S)

PIC_IDX = 500

# ---------- HELPERS ----------
def to_relative(st, en, t):
    P,T,_ = t.shape
    rel_st, rel_en = [], []
    for p in range(P):
        for tr in range(T):
            s, e = st[p, tr], en[p, tr]
            if np.isfinite(s) and np.isfinite(e):
                pic0 = t[p, tr, PIC_IDX]
                rel_st.append(s - pic0); rel_en.append(e - pic0)
    return np.asarray(rel_st, float), np.asarray(rel_en, float)

def describe(a):
    if a.size == 0: return dict(n=0, mean=np.nan, sd=np.nan, min=np.nan, max=np.nan)
    return dict(n=int(a.size), mean=float(np.mean(a)), sd=float(np.std(a, ddof=1)) if a.size>1 else np.nan,
                min=float(np.min(a)), max=float(np.max(a)))

def fd_bins(v):
    v=v[~np.isnan(v)]
    if v.size<2: return 'auto'
    q75,q25=np.percentile(v,[75,25]); iqr=q75-q25
    if iqr<=0: return 'auto'
    h=2*iqr*(v.size**(-1/3)); 
    return max(10,int(np.ceil((v.max()-v.min())/h))) if h>0 else 'auto'

def plot_hist(v, title, xlabel):
    v=v[~np.isnan(v)]
    if v.size==0: 
        print(f"[warn] No data for {title}"); return
    plt.figure(figsize=(6,3.6))
    plt.hist(v, bins=fd_bins(v), edgecolor="black", linewidth=0.7)
    plt.grid(True, alpha=0.35); plt.xlabel(xlabel); plt.ylabel("Count")
    plt.title(title); plt.tight_layout(); plt.show()

# ---------- FIRST ----------
fst, fen = to_relative(first_st, first_en, tvec)
fdur = fen - fst
sdesc, edesc, ddesc = describe(fst), describe(fen), describe(fdur)
print("\n=== FREE VIEWING — FIRST picture dwell (relative to picture onset, ms) ===")
print(f"Start: mean={sdesc['mean']:.2f} SD={sdesc['sd']:.2f} min={sdesc['min']:.2f} max={sdesc['max']:.2f} n={sdesc['n']}")
print(f"End:   mean={edesc['mean']:.2f} SD={edesc['sd']:.2f} min={edesc['min']:.2f} max={edesc['max']:.2f} n={edesc['n']}")
print(f"Dur:   mean={ddesc['mean']:.2f} SD={ddesc['sd']:.2f} min={ddesc['min']:.2f} max={ddesc['max']:.2f} n={ddesc['n']}")
plot_hist(fst,  "First dwell — start (ms rel.)", "Start (ms)")
plot_hist(fen,  "First dwell — end (ms rel.)",   "End (ms)")
plot_hist(fdur, "First dwell — duration (ms)",   "Duration (ms)")

# ---------- LAST ----------
lst, len_ = to_relative(last_st, last_en, tvec)
ldur = len_ - lst
sdesc, edesc, ddesc = describe(lst), describe(len_), describe(ldur)
print("\n=== FREE VIEWING — LAST picture dwell (relative to picture onset, ms) ===")
print(f"Start: mean={sdesc['mean']:.2f} SD={sdesc['sd']:.2f} min={sdesc['min']:.2f} max={sdesc['max']:.2f} n={sdesc['n']}")
print(f"End:   mean={edesc['mean']:.2f} SD={edesc['sd']:.2f} min={edesc['min']:.2f} max={edesc['max']:.2f} n={edesc['n']}")
print(f"Dur:   mean={ddesc['mean']:.2f} SD={ddesc['sd']:.2f} min={ddesc['min']:.2f} max={ddesc['max']:.2f} n={ddesc['n']}")
plot_hist(lst,  "Last dwell — start (ms rel.)", "Start (ms)")
plot_hist(len_, "Last dwell — end (ms rel.)",   "End (ms)")
plot_hist(ldur, "Last dwell — duration (ms)",   "Duration (ms)")
