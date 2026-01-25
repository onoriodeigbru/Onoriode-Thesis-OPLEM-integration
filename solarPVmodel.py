import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, List
import json
import requests
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

@dataclass
class PVParameters:
    eta_ref: float = 0.20          # Reference efficiency (20%)
    beta: float = 0.0045           # Temperature coefficient (1/°C)
    T_ref: float = 25.0            # Reference temperature (°C)
    area: float = 1.64             # Module area (m²)
    NOCT: float = 45.0             # Nominal Operating Cell Temperature (°C)
    rated_power: float = 300.0     # Rated power at STC (W)
    
    def __post_init__(self):
        
        assert 0 < self.eta_ref < 1, "Efficiency must be between 0 and 1"
        assert self.beta > 0, "Temperature coefficient must be positive"
        assert self.area > 0, "Area must be positive"


class SolarPVModel:
    
    def __init__(self, params: Optional[PVParameters] = None):
        self.params = params if params else PVParameters()
        self.simulation_results = None
    
    def calculate_cell_temperature(self, 
                                   T_ambient: float, 
                                   G_t: float,
                                   wind_speed: Optional[float] = None) -> float:
        
        if G_t <= 0:
            return T_ambient
        
        # NOCT-based correlation
        delta_T = (self.params.NOCT - 20) * (G_t / 800)
        T_cell = T_ambient + delta_T
        
        # Wind cooling correction
        if wind_speed is not None and wind_speed > 0:
            cooling_factor = 1 / (1 + 0.05 * wind_speed)
            T_cell = T_ambient + delta_T * cooling_factor
        
        return T_cell
    
    def calculate_power(self, G_t: float, T_cell: float) -> float:
   
        if G_t <= 0:
            return 0.0
        
        # Temperature derating factor
        temp_factor = 1 - self.params.beta * (T_cell - self.params.T_ref)
        temp_factor = max(0, temp_factor)
        
        # P_PV = η_ref * [1 - β(T_cell - T_ref)] * G_t * A
        P_PV = self.params.eta_ref * temp_factor * G_t * self.params.area
        
        return P_PV
    
    def get_efficiency(self, T_cell: float) -> float:
        """Calculate actual efficiency at given cell temperature."""
        temp_factor = 1 - self.params.beta * (T_cell - self.params.T_ref)
        return self.params.eta_ref * max(0, temp_factor)
    
    def simulate_timeseries(self,
                           irradiance: np.ndarray,
                           ambient_temp: np.ndarray,
                           wind_speed: Optional[np.ndarray] = None,
                           timestamps: Optional[np.ndarray] = None) -> pd.DataFrame:
      
        n = len(irradiance)
        assert len(ambient_temp) == n,
        
        if wind_speed is not None:
            assert len(wind_speed) == n, 
        
        power_output = np.zeros(n)
        cell_temps = np.zeros(n)
        efficiencies = np.zeros(n)
        
        for i in range(n):
            wind = wind_speed[i] if wind_speed is not None else None
            
            T_cell = self.calculate_cell_temperature(
                ambient_temp[i], irradiance[i], wind
            )
            cell_temps[i] = T_cell
            power_output[i] = self.calculate_power(irradiance[i], T_cell)
            efficiencies[i] = self.get_efficiency(T_cell)
        
        if timestamps is None:
            timestamps = np.arange(n)
        
        results = pd.DataFrame({
            'time': timestamps,
            'irradiance_W_m2': irradiance,
            'ambient_temp_C': ambient_temp,
            'cell_temp_C': cell_temps,
            'power_W': power_output,
            'efficiency': efficiencies
        })
        
        self.simulation_results = results
        return results

