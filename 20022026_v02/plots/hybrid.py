import matplotlib.pyplot as plt
import numpy as np
from plots.base import ieee_style, COLORS

def plot_hybrid_generation( pv_power, wind_power, save=None):
    ieee_style()
    fig, ax = plt.subplots()

    
    ax.plot( pv_power, color=COLORS["pv"], label="pv_power")
    ax.plot( wind_power, color=COLORS["wind"], label="wind_power")
    ax.plot( pv_power + wind_power, color=COLORS["total"], label="Total")

    ax.set_xlabel("Time (h)")
    ax.set_ylabel("Power (kW)")
    ax.legend()
    ax.grid(True)

    """
    if save:
        fig.savefig(save, bbox_inches="tight")
    return fig
    """
    plt.tight_layout()
    plt.show()

def plot_energy_share(pv_energy, wind_energy, save=None):
    ieee_style()
    fig, ax = plt.subplots()

    ax.pie(
        [pv_energy, wind_energy],
        labels=["pv_power", "wind_power"],
        colors=[COLORS["pv"], COLORS["wind"]],
        autopct="%.1f%%",
        startangle=90,
    )
    ax.set_title("Energy Contribution")
    """
    if save:
        fig.savefig(save, bbox_inches="tight")
    return fig
    """
    
    plt.tight_layout()
    plt.show()