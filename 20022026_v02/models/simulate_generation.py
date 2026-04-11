#---- import modules ----
import pandas as pd
from models.pv_model import *
from models.wind_model import *

sw = simulate_wind
spv = simulate_pv

#---- defining the simulation function ----
def simulate_system(data: pd.DataFrame) -> pd.DataFrame:

    #time_step_hours = data["time"].values

    # ----- Wind -----
    wind = sw(data["wind_speed"].values)

    # ----- PV -----
    pv = spv(
        irradiance=data["ghi"].values, 
        temp_air=data["temp_air"].values
        )

    # ----- Assemble results -----
    results = data.copy()

    results["wind_power"] = wind["wind_power"].values
    results["pv_power"] = pv["pv_power"].values
    results["total_power"] = ((results["wind_power"] + results["pv_power"]))
    results["pv_fraction"] = pv["pv_power"].values / (results["total_power"] + 1e-6)
    #results["total_energy_Wh"] = results["total_power"] * time_step_hours

    return results
    
