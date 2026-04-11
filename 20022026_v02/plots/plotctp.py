"""
---------------
Plots all market results from pickle files.
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import os
import warnings
warnings.filterwarnings('ignore')

# ── Colour scheme ──────────────────────────────────────────────────────────────
COLORS = {
    'Central': '#2196F3',   # blue
    'ToU':     '#4CAF50',   # green
    'P2P':     '#FF5722',   # orange
}
ASSET_COLORS = plt.cm.tab20.colors

# ── Helper ─────────────────────────────────────────────────────────────────────
def load(path):
    try:
        return pickle.load(open(path, "rb"))
    except Exception as e:
        print(f"  ⚠ Could not load {path}: {e}")
        return None

def hours(T=24):
    return np.arange(T)

os.makedirs("Plots", exist_ok=True)

# ==============================================================================
# 1. VOLTAGES
# ==============================================================================
print("Plotting voltages...")
fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
fig.suptitle("Voltage Profiles", fontsize=16, fontweight='bold')

markets = ['Central', 'ToU', 'P2P']
for ax, market in zip(axes, markets):
    v = load(f"Results/{market}/voltage.p")
    if v is None:
        ax.set_title(f"{market} — no data")
        continue
    v = np.array(v)
    T = v.shape[0]
    t = hours(T)

    ax.fill_between(t, v[:, 0], v[:, 1],
                    alpha=0.25, color=COLORS[market], label='Min–Max range')
    ax.plot(t, v[:, 0], color=COLORS[market], lw=1.5, ls='--', label='Vmin')
    ax.plot(t, v[:, 1], color=COLORS[market], lw=1.5,       label='Vmax')
    ax.axhline(1.05, color='red',  lw=1, ls=':', label='Upper limit (1.05)')
    ax.axhline(0.95, color='red',  lw=1, ls=':', label='Lower limit (0.95)')
    ax.set_title(market, fontsize=13, fontweight='bold')
    ax.set_xlabel("Time (h)")
    ax.set_xticks([0,8,16,23],('00:00', '08:00', '16:00', '23:00'))
    ax.set_ylabel("Voltage (p.u.)")
    ax.set_xlim(0, T-1)
    ax.legend(fontsize=8, loc='lower right')
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.legend()
plt.savefig("Plots/voltages.png", dpi=150, bbox_inches='tight')
plt.close()
print("  ✓ Plots/voltages.png")

# ==============================================================================
# 2. ASSETS' STATES  (Battery SoE + Indoor Temperature)
# ==============================================================================
print("Plotting assets' states...")
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle("Assets' States", fontsize=16, fontweight='bold')

for col, market in enumerate(markets):
    # --- Battery State of Energy ---
    ax = axes[0, col]
    eb = load(f"Results/{market}/Ebatt.p")
    if eb is not None:
        eb = np.array(eb)          # shape (N_batt, T)
        T  = eb.shape[1]
        t  = hours(T)
        for i in range(eb.shape[0]):
            ax.plot(t, eb[i], color=ASSET_COLORS[i % len(ASSET_COLORS)],
                    lw=1.2, alpha=0.8)
        ax.set_title(f"{market} — Battery SoE", fontsize=12, fontweight='bold')
        ax.set_xlabel("Time (h)")
        ax.set_xticks([0,8,16,23],('00:00', '08:00', '16:00', '23:00'))
        ax.set_ylabel("Energy (kWh)")
        ax.set_xlim(0, T-1)
        ax.grid(True, alpha=0.3)
    else:
        ax.set_title(f"{market} — Battery SoE (no data)")

    # --- Indoor Temperature ---
    ax = axes[1, col]
    tin = load(f"Results/{market}/Tin.p")
    if tin is not None:
        tin = np.array(tin)        # shape (N_build, T)
        T   = tin.shape[1]
        t   = hours(T)
        for i in range(tin.shape[0]):
            ax.plot(t, tin[i], color=ASSET_COLORS[i % len(ASSET_COLORS)],
                    lw=1.2, alpha=0.8)
        ax.set_title(f"{market} — Indoor Temperature", fontsize=12, fontweight='bold')
        ax.set_xlabel("Time (h)")
        ax.set_xticks([0,8,16,23],('00:00', '08:00', '16:00', '23:00'))
        ax.set_ylabel("Temperature (°C)")
        ax.set_xlim(0, T-1)
        ax.grid(True, alpha=0.3)
    else:
        ax.set_title(f"{market} — Temperature (no data)")

plt.tight_layout()
plt.legend ()

plt.savefig("Plots/assets_states.png", dpi=150, bbox_inches='tight')
plt.close()
print("  ✓ Plots/assets_states.png")

# ==============================================================================
# 3. EXCHANGES AND REVENUES
# ==============================================================================
print("Plotting exchanges and revenues...")

for market in markets:
    mc = load(f"Results/{market}/mc.p")
    if mc is None:
        print(f"  ⚠ {market}: no mc data")
        continue

    import pandas as pd
    if not isinstance(mc, pd.DataFrame):
        try:
            mc = pd.DataFrame(mc, columns=['time','seller','buyer','energy','price'])
        except Exception as e:
            print(f"  ⚠ {market}: could not parse mc — {e}")
            continue

    if mc.empty:
        print(f"  ⚠ {market}: mc is empty, skipping")
        continue

    mc['revenue'] = mc['energy'] * mc['price']
    T = int(mc['time'].max()) + 1
    t = np.arange(T)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"{market} — Exchanges & Revenues", fontsize=14, fontweight='bold')

    # Energy exchanged per timestep
    ax = axes[0]
    imports = mc[mc['seller'] == 0].groupby('time')['energy'].sum().reindex(t, fill_value=0)
    exports = mc[mc['buyer']  == 0].groupby('time')['energy'].sum().reindex(t, fill_value=0)
    ax.bar(t - 0.2, imports, 0.4, label='Import',  color=COLORS[market], alpha=0.8)
    ax.bar(t + 0.2, exports, 0.4, label='Export',  color=COLORS[market], alpha=0.4)
    ax.set_title("Energy Exchanges (kWh)", fontsize=12)
    ax.set_xlabel("Time (h)")
    ax.set_xticks([0,8,16,23],('00:00', '08:00', '16:00', '23:00'))
    ax.set_ylabel("Energy (kWh)")
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    # Revenue per participant
    ax = axes[1]
    rev_buy  = mc[mc['seller'] == 0].groupby('buyer')['revenue'].sum()
    rev_sell = mc[mc['buyer']  == 0].groupby('seller')['revenue'].sum()
    all_parts = sorted(set(rev_buy.index) | set(rev_sell.index))
    buy_vals  = [rev_buy.get(p, 0)  for p in all_parts]
    sell_vals = [rev_sell.get(p, 0) for p in all_parts]
    x = np.arange(len(all_parts))
    ax.bar(x - 0.2, buy_vals,  0.4, label='Cost (buying)',  color='#E53935', alpha=0.8)
    ax.bar(x + 0.2, sell_vals, 0.4, label='Revenue (selling)', color='#43A047', alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([f"P{p}" for p in all_parts], rotation=45, fontsize=7)
    ax.set_title("Revenue per Participant (£)", fontsize=12)
    ax.set_xlabel("Participant")
    ax.set_ylabel("£")
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.legend()
    plt.savefig(f"Plots/exchanges_{market}.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Plots/exchanges_{market}.png")

# ==============================================================================
# 4. SCHEDULES PER PARTICIPANT
# ==============================================================================
print("Plotting schedules per participant...")

for market in markets:
    raw = load(f"Results/{market}/schedules.p")
    if raw is None:
        print(f"  ⚠ {market}: no schedules data")
        continue

    # schedules is a list-of-lists (inhomogeneous): [participant][asset] = array(T,)
    if not isinstance(raw, (list, tuple)):
        print(f"  ⚠ {market}: unexpected schedules format")
        continue

    n_participants = min(4, len(raw))
    raw = raw[:n_participants]
    if n_participants == 0:
        print(f"  ⚠ {market}: empty schedules")
        continue

    ncols = min(2, n_participants)
    nrows = int(np.ceil(n_participants / ncols))
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(5*ncols, 3*nrows),
                             squeeze=False)
    fig.suptitle(f"{market} — Schedules per Participant",
                 fontsize=14, fontweight='bold')

    for p_idx, participant_schedules in enumerate(raw):
        row, col = divmod(p_idx, ncols)
        ax = axes[row][col]

        if participant_schedules is None or len(participant_schedules) == 0:
            ax.set_title(f"P{p_idx+1} — no data")
            ax.axis('off')
            continue

        for a_idx, sched in enumerate(participant_schedules):
            if sched is None:
                continue
            sched = np.array(sched).flatten()
            T = len(sched)
            ax.plot(hours(T), sched,
                    color=ASSET_COLORS[a_idx % len(ASSET_COLORS)],
                    lw=1.2, label=f"Asset {a_idx+1}")

        ax.axhline(0, color='black', lw=0.8, ls='--')
        ax.set_title(f"Participant {p_idx+1}", fontsize=10, fontweight='bold')
        ax.set_xlabel("Time (h)", fontsize=8)
        ax.set_xticks([0,8,16,23],('00:00', '08:00', '16:00', '23:00'))
        ax.set_ylabel("Power (kW)", fontsize=8)
        ax.grid(True, alpha=0.3)
        if len(participant_schedules) <= 6:
            ax.legend(fontsize=7)

    # Hide unused subplots
    for idx in range(n_participants, nrows * ncols):
        row, col = divmod(idx, ncols)
        axes[row][col].axis('off')

    plt.tight_layout()
    plt.legend()
    
    plt.savefig(f"Plots/schedules_{market}.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Plots/schedules_{market}.png")

print("\nDone! All plots saved to Plots/")