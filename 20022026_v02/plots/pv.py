import matplotlib.pyplot as plt
from plots.base import ieee_style, COLORS


def plot_pv_timeseries(ghi, power):
    ieee_style()
    fig, ax1 = plt.subplots()

    ax1.plot( 
                     ghi, 
                     color=COLORS["grid"],  
                     label="GHI (W/m²)",
                     linewidth=0.3
                     )
    
    ax1.set_ylabel("Irradiance (W/m²)")

    ax2 = ax1.twinx()
    ax2.plot(
             power, 
             color=COLORS["pv"], 
             label="PV Power (kW)",
             linewidth=0.3
             )
    
    ax2.set_ylabel("Power (kW)")

    ax1.set_xlabel("Time (h)")
    ax1.legend(loc="upper left")
    ax2.legend(loc="upper right")
    ax1.grid(True)

    plt.tight_layout()
    plt.show()
    
    #if save:
    #    fig.savefig(save, bbox_inches="tight")
    
    return fig


def plot_pv_power(pv_power):
    ################################
    #Plot PV power output over time.#
    #################################
    fig, ax = plt.subplots()

    ax.plot(
        pv_power,
        label="PV Power",
        linewidth=0.25
    )

    ax.set_xlabel("Time step")
    ax.set_ylabel("Power (W)")
    ax.set_title("PV Power Output")
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.tight_layout()
    plt.show()

    return fig

"""
import matplotlib.pyplot as plt
import numpy as np


def plot_pv_timeseries(time, ghi, power):
    ieee_style()
    fig, ax1 = plt.subplots(figsize=(10, 4))
    
    # GHI (left axis)

    ax1.fill_between(
        time,
        ghi,
        alpha=0.3,
        label="GHI (W/m²)"
    )
    ax1.set_ylabel("Irradiance (W/m²)")

    # Power (right axis)
    ax2 = ax1.twinx()
    ax2.plot(
        time,
        power,
        label="PV Power (kW)",
        linewidth=1.2
    )
    ax2.set_ylabel("Power (kW)")

    # Force 24-hour display
    ax1.set_xlim(0, 24)
    ax1.set_xticks(np.arange(0, 24, 1))  # every 2 hours
    ax1.set_xlabel("Time (hours)")

    # Legends
    ax1.legend(loc="upper left")
    ax2.legend(loc="upper right")

    ax1.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    return fig
"""