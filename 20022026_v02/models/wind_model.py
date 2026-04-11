import numpy as np
import pandas as pd

def simulate_wind(wind_speed: np.ndarray ,
                  rated_power_W: float = 2000.0,
                  cut_in: float = 3.0,
                  rated_speed: float = 12.0,
                  cut_out: float = 25.0) -> pd.DataFrame:

    """
    Wind turbine power curve (piecewise cubic).
    - P = 0                                          if V < V_cut_in
    - P = P_rated * ((V - V_cut_in)/(V_rated - V_cut_in))³  if V_cut_in ≤ V < V_rated
    - P = P_rated                                    if V_rated ≤ V < V_cut_out
    - P = 0                                          if V ≥ V_cut_out
    """
    v = np.asarray(wind_speed, dtype=float)
    wind_power = np.zeros_like(v)

    mask1 = (v >= cut_in) & (v < rated_speed)
    mask2 = (v >= rated_speed) & (v <= cut_out)

    #P = P_rated * ((V - V_cut_in)/(V_rated - V_cut_in))³  if V_cut_in ≤ V < V_rated
    wind_power[mask1] = rated_power_W * ((v[mask1] - cut_in)/(rated_speed - cut_in)) ** 3
    
    #P = P_rated                                    if V_rated ≤ V < V_cut_out
    wind_power[mask2] = rated_power_W


    #results
    results = pd.DataFrame({
        "wind_speed":v,
        "wind_power": wind_power
    })

    return results