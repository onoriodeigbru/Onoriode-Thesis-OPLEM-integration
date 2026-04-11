import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import os

os.makedirs("Plots", exist_ok=True)

def load(path):
    try:
        return pickle.load(open(path, "rb"))
    except Exception as e:
        print(f"  could not load {path}: {e}")
        return None

markets = ['ToU', 'P2P', 'Central']
colors  = ['#2196F3', '#FF9800', '#4CAF50']   # blue, orange, green

# ── DSO revenue ───────────────────────────────────────────────────────────────
# DSO buys from exporters (buyer=0) at export price and sells to importers (seller=0)
# Revenue = sum(price * energy) for seller=0 trades minus sum(price * energy) for buyer=0 trades
dso_revenue = []
total_exports = []

for market in markets:
    mc = load(f"Results/{market}/mc.p")
    if mc is None:
        dso_revenue.append(0)
        total_exports.append(0)
        continue

    # ensure numeric
    mc['price']  = mc['price'].astype(float)
    mc['energy'] = mc['energy'].astype(float)

    # DSO revenue: collects from imports, pays for exports
    imp = mc[mc['seller'] == 0]   # grid sells → participant imports
    exp = mc[mc['buyer']  == 0]   # participant sells → grid buys

    rev = (imp['price'] * imp['energy']).sum() - (exp['price'] * exp['energy']).sum()
    dso_revenue.append(rev)

    # total exports = energy sold back to grid
    total_exports.append(exp['energy'].sum())

# ── computation times ─────────────────────────────────────────────────────────
# Replace these with your actual recorded times if available
comp_times = [1.055, 972.09, 545.94]   # ToU, P2P, Central (seconds)

# ── figure ────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(13, 5))
fig.suptitle("DSO revenue, total exports and computation times",
             fontsize=13, fontweight='bold', y=1.02)

def bar_panel(ax, values, ylabel, title, letter, fmt='.4f'):
    bars = ax.bar(markets, values, color=colors, width=0.5, edgecolor='white', linewidth=0.8)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + max(values)*0.01,
                f"{val:{fmt}}",
                ha='center', va='bottom', fontsize=9, fontweight='bold')
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_title(title, fontsize=10, fontweight='bold')
    ax.set_xlabel(f"({letter})", fontsize=10)
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
    ax.grid(True, axis='y', alpha=0.3, linewidth=0.7)
    ax.spines[['top', 'right']].set_visible(False)
    ax.set_ylim(0, max(values) * 1.18)

bar_panel(axes[0], dso_revenue,  "DSO revenue (£)",        "DSO revenue",        "a", fmt='.4f')
bar_panel(axes[1], total_exports,"Total exports (kWh)",    "Total exports",      "b", fmt='.3f')
bar_panel(axes[2], comp_times,   "Computation time (sec)", "Computation time",   "c", fmt='.2f')

plt.tight_layout()
out = "Plots/dso_summary.png"
plt.savefig(out, dpi=150, bbox_inches='tight')
plt.close()
print(f"✓ saved {out}")

print(f"\nSummary:")
for i, m in enumerate(markets):
    print(f"  {m}: DSO revenue={dso_revenue[i]:.4f} £  |  exports={total_exports[i]:.3f} kWh  |  time={comp_times[i]:.2f} s")