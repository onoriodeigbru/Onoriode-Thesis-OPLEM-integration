import numpy as np
import pandas as pd
from typing import Optional, Dict, Union, NamedTuple, Tuple
from model import SolarPVModel, WindTurbineModel

class ModelValidator:
   
    MetricValue = Union[float, int]

    @staticmethod
    def calculate_metrics(measured: np.ndarray, 
                         simulated: np.ndarray) -> Dict[str, MetricValue]:

        measured = np.asarray(measured, dtype=float)
        simulated = np.asarray(simulated, dtype=float)

        if measured.shape != simulated.shape:
            raise ValueError("Measured and simulated arrays must have same shape")

        # Remove NaNs
        valid = ~np.isnan(measured) & ~np.isnan(simulated)
        measured = measured[valid]
        simulated = simulated[valid]

        if measured.size == 0:
            raise ValueError("No valid samples for metric calculation")
        
        # Calculate errors
        errors = simulated - measured
        abs_errors = np.abs(errors)
        
        # Mean Absolute Error (MAE)
        mae = np.mean(abs_errors)
        
        # Root Mean Square Error (RMSE)
        rmse = np.sqrt(np.mean(errors**2))
        
        # Mean Absolute Percentage Error (MAPE)
        positive = measured > 0
        if np.any(positive):
            mape = np.mean(abs_errors[positive] / measured[positive]) * 100
        else:
            mape = np.nan 
        
        # R-squared (Coefficient of Determination)
        ss_tot = np.sum((measured - np.mean(measured)) ** 2)
        if ss_tot > 0:
            ss_res = np.sum(errors ** 2)
            r_squared = 1 - ss_res / ss_tot
        else:
            r_squared = np.nan

        return {
        'MAE': float(mae),
        'RMSE': float(rmse),
        'MAPE': float(mape),
        'R²': float(r_squared),
        'max_error': float(np.max(abs_errors)),
        'mean_error': float(np.mean(errors)),
        'n_samples': float(measured.size)
         }

    @staticmethod
    def calculate_performance_metrics(
        results: pd.DataFrame, 
        time_step_hours: float = 1.0,
        rated_power: float = 300.0) -> Dict [str, float]:

        power = np.asarray(results['power_W'], dtype=float)
        total_energy_Wh = power.sum() * time_step_hours
        
        return {
            'total_energy_kWh': total_energy_Wh / 1000,
            'peak_power_W': power.max(),
            'average_power_W': power.mean(),
            'capacity_factor': power.mean() / rated_power if rated_power > 0 else 0
        }


class PVModelValidator:
    
    def __init__(self, pv_model: SolarPVModel):
        self.pv_model = pv_model
    
    def prepare_nrel_data(
            self, nrel_df: pd.DataFrame
            ) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        
        irradiance = np.asarray(
            nrel_df['GHI'] if 'GHI' in nrel_df else nrel_df['ghi'],
            dtype=float
        )
        
        temperature = np.asarray(
            nrel_df['Temperature'] if 'Temperature' in nrel_df else nrel_df['air_temperature'],
            dtype=float
        )
        
        if 'Wind Speed' in nrel_df:
            wind_speed = np.asarray(nrel_df['Wind Speed'], dtype=float)
        elif 'wind_speed' in nrel_df:
            wind_speed = np.asarray(nrel_df['wind_speed'], dtype=float)
        else:
            wind_speed = None
        
        return irradiance, temperature, wind_speed
    
    def prepare_pvgis_data(self, pvgis_df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """Extract irradiance, temperature, wind from PVGIS DataFrame."""
        # PVGIS uses G(i) for irradiance on inclined plane
        irradiance = np.asarray(
            pvgis_df['G(i)'] if 'G(i)' in pvgis_df else pvgis_df['Gb(i)'],
            dtype=float
        )
        
        temperature = np.asarray(
            pvgis_df['T2m'] if 'T2m' in pvgis_df else pvgis_df['air_temperature'],
            dtype=float
        )
        
        if 'WS10m' in pvgis_df:
            wind_speed = np.asarray(pvgis_df['WS10m'], dtype=float)
        elif 'WS10m' in pvgis_df:
            wind_speed = np.asarray(pvgis_df['WS10m'], dtype=float)
        else:
            wind_speed = None

        
        return irradiance, temperature, wind_speed
    
    def validate_with_nrel(self, 
                          nrel_df: pd.DataFrame,
                          sample_size: Optional[int] = None) -> Dict:
        """Validate PV model using NREL data."""
        print("\n" + "="*70)
        print("PV MODEL VALIDATION WITH NREL DATA")
        print("="*70)
        
        irradiance, temperature, wind_speed = self.prepare_nrel_data(nrel_df)
        
        if sample_size:
            irradiance = irradiance[:sample_size]
            temperature = temperature[:sample_size]
            if wind_speed is not None:
                wind_speed = wind_speed[:sample_size]
        
        results = self.pv_model.simulate_timeseries(
            irradiance=irradiance,
            ambient_temp=temperature,
            wind_speed=wind_speed
        )
        
        metrics = ModelValidator.calculate_performance_metrics(
            results=results, 
            rated_power=self.pv_model.params.rated_power
        )
        
        print(f"\n✓ Processed {len(results)} records")
        print(f"  Total Energy: {metrics['total_energy_kWh']:.2f} kWh")
        print(f"  Peak Power: {metrics['peak_power_W']:.1f} W")
        print(f"  Capacity Factor: {metrics['capacity_factor']*100:.2f}%")
        
        return {'results': results, 'metrics': metrics, 'source': 'NREL'}
    
    def validate_with_pvgis(self, pvgis_df: pd.DataFrame) -> Dict:
        """Validate PV model using PVGIS data."""
        print("\n" + "="*70)
        print("PV MODEL VALIDATION WITH PVGIS DATA")
        print("="*70)
        
        irradiance, temperature, wind_speed = self.prepare_pvgis_data(pvgis_df)
        
        results = self.pv_model.simulate_timeseries(
            irradiance=irradiance,
            ambient_temp=temperature,
            wind_speed=wind_speed
        )
        
        metrics = ModelValidator.calculate_performance_metrics(
            results, rated_power=self.pv_model.params.rated_power
        )
        
        print(f"\n✓ Processed {len(results)} records")
        print(f"  Total Energy: {metrics['total_energy_kWh']:.2f} kWh")
        print(f"  Peak Power: {metrics['peak_power_W']:.1f} W")
        
        return {'results': results, 'metrics': metrics, 'source': 'PVGIS'}


class WindModelValidator:
    """Validator specifically for Wind Turbine models."""
    
    def __init__(self, wind_model: WindTurbineModel):
        self.wind_model = wind_model
    
    def prepare_nrel_data(self, nrel_df: pd.DataFrame) -> Optional[np.ndarray]:
        """Extract wind speed from NREL DataFrame."""
        if 'Wind Speed' not in nrel_df:
            return None
        
        return np.asarray(nrel_df['Wind Speed'].values)
    
    def prepare_pvgis_data(
            self, pvgis_df: pd.DataFrame) -> Optional[np.ndarray]:
        """Extract wind speed from PVGIS DataFrame."""
        if 'WS10m' not in pvgis_df:
            return None

        return np.asarray(pvgis_df['WS10m'], dtype=float)

    
    def validate_with_data(self, 
                          wind_speed: np.ndarray,
                          source: str = "Data") -> Dict:
        """Validate wind model with wind speed data."""
        print(f"\n[WIND] Validating with {source}...")
        
        results = self.wind_model.simulate_timeseries(
            wind_speed=wind_speed,
            adjust_height=True
        )
        
        metrics = ModelValidator.calculate_performance_metrics(
            results, rated_power=self.wind_model.params.P_rated
        )
        
        print(f"✓ Processed {len(results)} records")
        print(f"  Total Energy: {metrics['total_energy_kWh']:.2f} kWh")
        print(f"  Capacity Factor: {metrics['capacity_factor']*100:.2f}%")
        
        return {'results': results, 'metrics': metrics, 'source': source}

class WeatherData(NamedTuple):
    irradiance: np.ndarray
    temperature: np.ndarray
    wind_speed: Optional[np.ndarray]

class RenewableEnergySystem:
    """
    Integrated system combining Solar PV and Wind generation.
    for LEM simulation with OPLEM.
    """
    
    def __init__(self, 
                 pv_model: SolarPVModel,
                 wind_model: WindTurbineModel,
                 n_pv_panels: int = 10,
                 n_wind_turbines: int = 2):
    
        self.pv_model = pv_model
        self.wind_model = wind_model
        self.n_pv_panels = n_pv_panels
        self.n_wind_turbines = n_wind_turbines
    
    def simulate_system(self,
                       irradiance: np.ndarray,
                       ambient_temp: np.ndarray,
                       wind_speed: np.ndarray,
                       timestamps: Optional[np.ndarray] = None) -> pd.DataFrame:

        # Simulate PV
        pv_results = self.pv_model.simulate_timeseries(
            irradiance, ambient_temp, wind_speed, timestamps
        )
        
        # Simulate Wind
        wind_results = self.wind_model.simulate_timeseries(
            wind_speed, timestamps, adjust_height=True
        )
        
        # Combine results
        total_pv_power = pv_results['power_W'] * self.n_pv_panels
        total_wind_power = wind_results['power_W'] * self.n_wind_turbines
        total_power = total_pv_power + total_wind_power
        
        results = pd.DataFrame({
            'time': pv_results['time'],
            'irradiance_W_m2': irradiance,
            'ambient_temp_C': ambient_temp,
            'wind_speed_m_s': wind_speed,
            'pv_power_W': total_pv_power,
            'wind_power_W': total_wind_power,
            'total_power_W': total_power,
            'pv_fraction': total_pv_power / (total_power + 1e-6)
        })
        
        return results
    
    def get_system_capacity(self) -> Dict:

        return {
            'pv_capacity_W': self.pv_model.params.rated_power * self.n_pv_panels,
            'wind_capacity_W': self.wind_model.params.P_rated * self.n_wind_turbines,
            'total_capacity_W': (self.pv_model.params.rated_power * self.n_pv_panels + 
                               self.wind_model.params.P_rated * self.n_wind_turbines)
        }
