import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt

# --- Paths (match your Cue saver) ---
ROOT = r"C:/Users/charl/OneDrive/Dokumente/Uni/7. Semester/Bachelor Thesis/ArrowTask"
MAT = lambda name: f"{ROOT}/{name}"

st_rel = sio.loadmat(MAT("fixationCrossStartTimes_LOCAL.mat"))["fixationStartTimes"]  
en_rel = sio.loadmat(MAT("fixationCrossEndTimes_LOCAL.mat"))  ["fixationEndTimes"]    


st = st_rel.ravel().astype(float)
en = en_rel.ravel().astype(float)
mask = np.isfinite(st) & np.isfinite(en) & (en >= st)
st, en = st[mask], en[mask]
du = en - st

def describe(a):
    if a.size == 0: return dict(mean=np.nan, sd=np.nan, min=np.nan, max=np.nan, n=0)
    return dict(mean=float(np.mean(a)), sd=float(np.std(a, ddof=1)) if a.size>1 else np.nan,
                min=float(np.min(a)), max=float(np.max(a)), n=int(a.size))

def fd_bins(v):
    v = np.asarray(v); 
    if v.size < 2: return 'auto'
    iqr = np.subtract(*np.percentile(v, [75,25])); 
    if iqr <= 0: return 'auto'
    h = 2*iqr*(v.size**(-1/3)); 
    if h <= 0: return 'auto'
    return max(int(np.ceil((v.max()-v.min())/h)), 10)

def plot_hist(vals, title, xlabel):
    v = np.asarray(vals, float)
    v = v[~np.isnan(v)]
    if v.size == 0:
        print(f"[warn] No data for: {title}")
        return
    plt.figure(figsize=(6,3.6))
    plt.hist(v, bins=fd_bins(v), edgecolor="black", linewidth=0.7)
    plt.grid(True, alpha=0.35)
    plt.xlabel(xlabel); plt.ylabel("Count")
    plt.title(title)
    plt.tight_layout(); plt.show()

# --- Print stats ---
st_desc = describe(st)
en_desc = describe(en)
du_desc = describe(du)

print("\n=== Cue Task — Cross dwell (relative to cross onset) ===")
print(f"Start: mean={st_desc['mean']:.2f}  SD={st_desc['sd']:.2f}  min={st_desc['min']:.0f}  max={st_desc['max']:.0f}  n={st_desc['n']}")
print(f"End:   mean={en_desc['mean']:.2f}  SD={en_desc['sd']:.2f}  min={en_desc['min']:.0f}  max={en_desc['max']:.0f}  n={en_desc['n']}")
print(f"Dur:   mean={du_desc['mean']:.2f}  SD={du_desc['sd']:.2f}  min={du_desc['min']:.0f}  max={du_desc['max']:.0f}  n={du_desc['n']}")

# --- Plots (same style as your snippet) ---
plot_hist(st, "Distribution of cross dwell START (relative to cross onset)", "Start (ms)")
plot_hist(en, "Distribution of cross dwell END (relative to cross onset)",   "End (ms)")
plot_hist(du, "Distribution of cross dwell DURATION",                        "Duration (ms)")
