import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional, Dict
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
        
        temp_factor = 1 - self.params.beta * (T_cell - self.params.T_ref)
        temp_factor = max(0, temp_factor)
        
        P_PV = self.params.eta_ref * temp_factor * G_t * self.params.area
        
        return P_PV
    
    def get_efficiency(self, T_cell: float) -> float:

        temp_factor = 1 - self.params.beta * (T_cell - self.params.T_ref)
        return self.params.eta_ref * max(0, temp_factor)
    
    def simulate_timeseries(self,
                           irradiance: np.ndarray,
                           ambient_temp: np.ndarray,
                           wind_speed: Optional[np.ndarray] = None,
                           timestamps: Optional[np.ndarray] = None) -> pd.DataFrame:

        n = len(irradiance)
        assert len(ambient_temp) == n, "Temperature profile length mismatch"
        
        if wind_speed is not None:
            assert len(wind_speed) == n,"Wind speed profile length mismatch"
        
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
    
    def calculate_energy(self, results: pd.DataFrame, time_step_hours: float = 1.0) -> float:

        total_energy_Wh = results['power_W'].sum() * time_step_hours
        return total_energy_Wh / 1000  # Convert to kWh
    
    def get_performance_metrics(self, results: pd.DataFrame, time_step_hours=1.0) -> Dict:
        
        total_energy_Wh = results['power_W'].sum() * time_step_hours
        max_possible_Wh = self.params.rated_power*len(results)*time_step_hours

        return {
            'total_energy_kWh': self.calculate_energy(results, time_step_hours),
            'peak_power_W': results['power_W'].max(),
            'average_power_W': results['power_W'].mean(),
            'max_cell_temp_C': results['cell_temp_C'].max(),
            'min_efficiency': results['efficiency'].min(),
            'avg_efficiency': results['efficiency'].mean(),
            'capacity_factor': total_energy_Wh / max_possible_Wh
        }
    
    def validate_with_measured_data(self, 
                                   measured_power: np.ndarray,
                                   simulated_power: np.ndarray) -> Dict:

        # Calculate errors
        errors = simulated_power - measured_power
        abs_errors = np.abs(errors)
        relative_errors = abs_errors / (measured_power + 1e-6) * 100  # Percentage
        
        # Calculate metrics
        mae = np.mean(abs_errors)  # Mean Absolute Error
        rmse = np.sqrt(np.mean(errors**2))  # Root Mean Square Error
        mape = np.mean(relative_errors)  # Mean Absolute Percentage Error
        
        # R-squared
        ss_res = np.sum(errors**2)
        ss_tot = np.sum((measured_power - np.mean(measured_power))**2)
        r_squared = 1 - (ss_res / (ss_tot + 1e-6))
        
        return {
            'MAE': mae,
            'RMSE': rmse,
            'MAPE': mape,
            'R2': r_squared,
            'max_error': np.max(abs_errors),
            'n_samples': len(measured_power)
        }
    
@dataclass
class WindTurbineParameters:

    P_rated: float = 2000.0        # Rated power (W)
    V_cut_in: float = 3.0          # Cut-in wind speed (m/s)
    V_rated: float = 12.0          # Rated wind speed (m/s)
    V_cut_out: float = 25.0        # Cut-out wind speed (m/s)
    hub_height: float = 10.0       # Hub height (m)
    rotor_diameter: float = 3.0    # Rotor diameter (m)
    
    def __post_init__(self):

        assert self.V_cut_in < self.V_rated < self.V_cut_out
        assert self.P_rated > 0
        assert self.hub_height > 0

class WindTurbineModel:

    def __init__(self, params: Optional[WindTurbineParameters] = None):
        self.params = params if params else WindTurbineParameters()
        self.simulation_results = None
    
    def adjust_wind_speed_to_hub_height(self, 
                                        V_measured: float,
                                        h_measured: float = 10.0,
                                        alpha: float = 0.143) -> float:

        V_hub = V_measured * (self.params.hub_height / h_measured) ** alpha
        return V_hub
    
    def calculate_power(self, V: float, adjust_height: bool = False) -> float:

        # Adjust to hub height if needed
        if adjust_height:
            V = self.adjust_wind_speed_to_hub_height(V)
        
        # Piecewise power curve
        if V < self.params.V_cut_in:
            return 0.0
        
        elif V < self.params.V_rated:
            # Cubic relationship in transition region
            ratio = (V - self.params.V_cut_in) / (self.params.V_rated - self.params.V_cut_in)
            return self.params.P_rated * (ratio ** 3)
        
        elif V < self.params.V_cut_out:
            return self.params.P_rated
        
        else:  # V >= V_cut_out
            return 0.0
    
    def simulate_timeseries(self,
                           wind_speed: np.ndarray,
                           timestamps: Optional[np.ndarray] = None,
                           adjust_height: bool = False) -> pd.DataFrame:
        n = len(wind_speed)
        power_output = np.zeros(n)
        
        for i in range(n):
            power_output[i] = self.calculate_power(wind_speed[i], adjust_height)
        
        if timestamps is None:
            timestamps = np.arange(n)
        
        results = pd.DataFrame({
            'time': timestamps,
            'wind_speed_m_s': wind_speed,
            'power_W': power_output
        })
        
        self.simulation_results = results
        return results
    
    def get_power_curve(self, V_range: Optional[np.ndarray] = None) -> pd.DataFrame:
        if V_range is None:
            V_range = np.linspace(0, self.params.V_cut_out + 5, 100)
        
        power = np.array([self.calculate_power(v) for v in V_range])
        
        return pd.DataFrame({
            'wind_speed_m_s': V_range,
            'power_W': power
        })
