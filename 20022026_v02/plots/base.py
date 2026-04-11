import matplotlib.pyplot as plt

def ieee_style():
    plt.rcParams.update({
        "figure.dpi": 300,
        "font.family": "serif",
        "font.size": 3,
        "axes.labelsize": 5,
        "axes.titlesize": 5,
        "legend.fontsize": 3,
        "xtick.labelsize": 2,
        "ytick.labelsize": 2,
        "lines.linewidth": 1.5,
        "grid.alpha": 0.3,
    })

COLORS = {
    "pv": "#0E8925",
    "wind": "#069BF1",
    "storage": "#CFB51F",
    "grid": "#840A0A",
    "total": "#000000",
}
