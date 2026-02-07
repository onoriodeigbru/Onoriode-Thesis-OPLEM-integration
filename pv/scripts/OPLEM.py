import pandas as pd
import json
from validation import RenewableEnergySystem



class OPLEMInterface:
    """
    Interface for integrating renewable models with OPLEM platform.
    Prepares data in format suitable for OPLEM API.
    """
    
    def __init__(self, renewable_system: RenewableEnergySystem):
        self.system = renewable_system
    
    def prepare_generation_profile(self, 
                                   simulation_results: pd.DataFrame,
                                   time_step_hours: float = 1.0) -> dict [str, float]:
        """
        Prepare renewable generation profile for OPLEM.
        
        Args:
            simulation_results: DataFrame from simulate_system
            time_step_hours: Time step duration
            
        Returns:
            Dictionary formatted for OPLEM API
        """
        profile = {
            'timestamp': simulation_results['time'].tolist(),
            'pv_generation_kW': (simulation_results['pv_power_W'] / 1000).tolist(),
            'wind_generation_kW': (simulation_results['wind_power_W'] / 1000).tolist(),
            'total_generation_kW': (simulation_results['total_power_W'] / 1000).tolist(),
            'time_step_hours': time_step_hours,
            'total_energy_kWh': simulation_results['total_power_W'].sum() * time_step_hours / 1000
        }
        
        return profile
    
    def export_to_json(self, profile: dict [str, float], filepath: str):
        """Export generation profile to JSON file for OPLEM."""
        with open(filepath, 'w') as f:
            json.dump(profile, f, indent=2)
        print(f"[OPLEM] ✓ Exported profile to {filepath}")
    
    def export_to_csv(self, simulation_results: pd.DataFrame, filepath: str):
        """Export simulation results to CSV for OPLEM."""
        # Convert to kW for standard format
        export_df = simulation_results.copy()
        export_df['pv_power_kW'] = export_df['pv_power_W'] / 1000
        export_df['wind_power_kW'] = export_df['wind_power_W'] / 1000
        export_df['total_power_kW'] = export_df['total_power_W'] / 1000
        
        export_df.to_csv(filepath, index=False)
        print(f"[OPLEM] ✓ Exported results to {filepath}")