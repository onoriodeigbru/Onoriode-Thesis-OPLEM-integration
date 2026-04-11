import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.figure as Figure
from plots.base import ieee_style, COLORS



def plot_wind_power_curve (wind_speed, 
                           wind_power, 
                           #save='wind_power_curve.png'
                           ):
    ieee_style()
    fig, ax = plt.subplots()

    ax.plot(wind_speed,
            wind_power,
            color=COLORS["wind"],
            linewidth=0.15
            )
    
    ax.set_xlabel("Wind Speed (m/s)")
    ax.set_ylabel("Power (kW)")
    ax.set_title("Wind Turbine Power Curve")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    #if save:
    #    fig.savefig(save, bbox_inches="tight")
    
    return fig


def plot_wind_timeseries( 
                         wind_speed, 
                         wind_power, 
                         #save='wind_timeseries.png'
                         ):
    ieee_style()
    fig, ax1 = plt.subplots()

    ax1.plot( 
             wind_speed, 
             color=COLORS["wind"], 
             alpha=0.3, 
             label="Wind Speed",
             linewidth=0.15)
    
    ax1.set_ylabel("Wind Speed (m/s)")

    ax2 = ax1.twinx()
    ax2.plot(
             wind_power,
             color=COLORS["wind"],
             label="Wind Power (kW)",
             linewidth=0.8)
    
    ax2.set_ylabel("Power (kW)")

    ax1.set_xlabel("Time (h)")
    ax1.legend(loc="upper left")
    ax2.legend(loc="upper right")
    ax1.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    #if save:
    #    fig.savefig(save, bbox_inches="tight")
    
    return fig

def plot_wind_power(wind_power):
    """
    Plot wind power output over time.
    """
    fig, ax = plt.subplots()
    
    ax.plot(wind_power,
            label="Wind Power (W)",
            linewidth=0.25
            )

    ax.set_xlabel("Time step")
    ax.set_ylabel("Power (W)")
    ax.set_title("Wind Power Output")
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.tight_layout()
    plt.show()

    return fig

