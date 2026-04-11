#import numpy and pandas libraries
import numpy as np
import pandas as pd

#Solar PV panel specifications and constants.

def simulate_pv(
        irradiance: np.ndarray,
        temp_air: np.ndarray,
        rated_power: float = 300.0,
        temp_coeff: float = -0.004,
        ref_temperature: float = 25.0,
        min_irradiance: float = 50.0
        ) -> pd.DataFrame:
        """
        Simulate PV power output (simple linear model: irradiance + temperature derating).
        
        Calculate PV power output.
        
        P_PV = η_ref * [1 - β(T_cell - T_ref)] * G_t * A
        
        Args:
        G: Solar irradiance (W/m²)
        T_cell: Cell temperature (°C)
            
        Returns:
            Power output (W)
        """

        G = np.asarray(irradiance, dtype=float)
        T = np.asarray(temp_air, dtype=float)

        #to prevent division/noise at very low irradiance
        G_eff = np.maximum(G, min_irradiance)
        
        #temperature factor
        temp_factor = 1 + temp_coeff * (T - ref_temperature)
        
        #PV power model
        pv = rated_power * (G_eff / 1000.0) * temp_factor

        #Clip negative values
        pv = np.clip (pv, 0.0, rated_power)

        results = pd.DataFrame({
            "ghi": G,
            "temp_air": T,
            "pv_power": pv
            })
        
        return results