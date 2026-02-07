import pandas as pd
from typing import Optional
import requests
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

class PVGISDataFetcher:

    def __init__(self):
        self.base_url = "https://re.jrc.ec.europa.eu/api/v5_2"
    
    def fetch_hourly_data(self,
                         latitude: float,
                         longitude: float,
                         year: int = 2020,
                         startyear: Optional[int] = None,
                         endyear: Optional[int] = None) -> Optional[pd.DataFrame]:

        if startyear is None:
            startyear = year
        if endyear is None:
            endyear = year
        
        params = {
            'lat': latitude,
            'lon': longitude,
            'startyear': startyear,
            'endyear': endyear,
            'outputformat': 'json',
            'pvcalculation': 0  # Radiation data only
        }
        
        endpoint = f"{self.base_url}/seriescalc"
        print(f"[PVGIS] Fetching data for ({latitude}, {longitude}) year {startyear}-{endyear}...")
        
        try:
            response = requests.get(endpoint, params=params, timeout=30)
            response.raise_for_status()
            data_json = response.json()
            
            hourly_data = data_json['outputs']['hourly']
            df = pd.DataFrame(hourly_data)
            
            print(f"[PVGIS] ✓ Successfully fetched {len(df)} records")
            return df
            
        except Exception as e:
            print(f"[PVGIS] ✗ Error: {e}")
            return None