import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.colors as mcolors
import os

# ── paths ────────────────────────────────────────────────────────────────────
RESULTS = "Results/Central"
PLOTS   = "Plots"
os.makedirs(PLOTS, exist_ok=True)

# ── load data ────────────────────────────────────────────────────────────────
def load(path):
    try:
        return pickle.load(open(path, "rb"))
    except Exception as e:
        print(f"  ⚠ could not load {path}: {e}")
        return None

info      = load(f"{RESULTS}/info.p")
dlmp      = np.load(f"{RESULTS}/dlmp_array.npy")   # (24, 55)
mc        = load(f"{RESULTS}/mc.p")

# ── price arrays ─────────────────────────────────────────────────────────────
# price_imp and price_exp are stored in the test script as numpy arrays.
# If you saved them separately, load them here instead.
# Otherwise we reconstruct a representative ToU profile from the mc data.
T   = dlmp.shape[0]          # 24
hrs = np.arange(T)

# Try to infer import/export prices from mc (seller=0 → import, buyer=0 → export)
if mc is not None:
    imp_rows = mc[mc['seller'] == 0]
    exp_rows = mc[mc['buyer']  == 0]
    price_imp_ts = imp_rows.groupby('time')['price'].mean().reindex(hrs).fillna(method='ffill').fillna(0).values
    price_exp_ts = exp_rows.groupby('time')['price'].mean().reindex(hrs).fillna(method='ffill').fillna(0).values
else:
    price_imp_ts = np.full(T, 0.06)
    price_exp_ts = np.full(T, 0.03)

# ── derived DLMP stats ────────────────────────────────────────────────────────
dlmp_mean = dlmp.mean(axis=1)   # (24,)
dlmp_min  = dlmp.min(axis=1)
dlmp_max  = dlmp.max(axis=1)
dlmp_p25  = np.percentile(dlmp, 25, axis=1)
dlmp_p75  = np.percentile(dlmp, 75, axis=1)

# ── colours ───────────────────────────────────────────────────────────────────
C_IMP  = "#378ADD"
C_EXP  = "#D85A30"
C_DLMP = "#1D9E75"
C_BAND = "#7F77DD"

# ── figure layout ─────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(14, 10))
fig.suptitle("CED Market — Aggregator Price Summary", fontsize=14,
             fontweight='bold', y=0.98)

gs = gridspec.GridSpec(3, 1, figure=fig, hspace=0.45,
                       height_ratios=[2.2, 1.6, 1.2])

# ── panel 1 : import / export / DLMP band ─────────────────────────────────────
ax1 = fig.add_subplot(gs[0])
ax1.fill_between(hrs, dlmp_min, dlmp_max, color=C_BAND, alpha=0.15,
                 label="DLMP range (min–max)")
ax1.plot(hrs, dlmp_p25, dlmp_p75, color='black', alpha=0.25,
                 label="DLMP IQR (25–75 %)")
ax1.plot(hrs, dlmp_mean,  color=C_DLMP, lw=2.2,   ls='--', label="Avg DLMP")
ax1.plot(hrs, price_imp_ts, color=C_IMP,  lw=2.2, label="Import price")
ax1.plot(hrs, price_exp_ts, color=C_EXP,  lw=2.2, label="Export price")

ax1.set_ylabel("Price (£/kWh)", fontsize=10)
ax1.set_xlabel("Time (h)", fontsize=8)
ax1.set_xticks([0,8,16,23],('00:00', '08:00', '16:00', '23:00'))
ax1.legend(fontsize=8, loc='right', ncol=1)
ax1.grid(True, alpha=0.25)
ax1.set_title("Import price · Export price · DLMP (all buses)", fontsize=11)

# ── panel 2 : DLMP heatmap (buses × time) ─────────────────────────────────────
ax2 = fig.add_subplot(gs[1])
vmin, vmax = np.percentile(dlmp, 2), np.percentile(dlmp, 98)
im = ax2.imshow(dlmp.T, aspect='auto', origin='lower',
                cmap='RdYlGn_r', vmin=vmin, vmax=vmax,
                extent=[-0.5, T-0.5, -0.5, dlmp.shape[1]-0.5])
cb = fig.colorbar(im, ax=ax2, fraction=0.02, pad=0.02)
cb.set_label("DLMP (£/kWh)", fontsize=9)
ax2.set_xlabel("Time (h)", fontsize=8)
ax2.set_ylabel("Bus index", fontsize=10)
ax2.set_xticks([0,8,16,23],('00:00', '08:00', '16:00', '23:00'))
ax2.set_title("DLMP heatmap — buses × time", fontsize=11)

# ── panel 3 : DLMP spread bar ─────────────────────────────────────────────────
ax3 = fig.add_subplot(gs[2])
spread = dlmp_max - dlmp_min
ax3.bar(hrs, spread, color=C_BAND, alpha=0.75, width=0.7)
ax3.axhline(spread.mean(), color=C_DLMP, lw=1.5, ls='--',
            label=f"Mean spread = {spread.mean():.4f} £/kWh")
ax3.set_xlabel("Time (h)", fontsize=8)
ax3.set_ylabel("DLMP spread (£/kWh)", fontsize=10)
ax3.set_xticks([0,8,16,23],('00:00', '08:00', '16:00', '23:00'))
ax3.legend(fontsize=8)
ax3.grid(True, alpha=0.25, axis='y')
ax3.set_title("DLMP spread across buses per timestep", fontsize=11)

# ── summary text box ──────────────────────────────────────────────────────────
summary = (
    f"Avg import price : {price_imp_ts.mean():.4f} £/kWh\n"
    f"Avg export price : {price_exp_ts.mean():.4f} £/kWh\n"
    f"Avg DLMP         : {dlmp_mean.mean():.4f} £/kWh\n"
    f"DLMP buses       : {dlmp.shape[1]}"
)
fig.text(0.01, 0.01, summary, fontsize=8,
         va='bottom', ha='left',
         bbox=dict(boxstyle='round,pad=0.4', fc='#f5f5f2', ec='#cccccc', alpha=0.8))

out = f"{PLOTS}/aggregator_Central.png"
plt.savefig(out, dpi=150, bbox_inches='tight')
plt.close()
print(f"✓ saved {out}")