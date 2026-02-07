import pandas as pd
from typing import Optional
import requests
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

class NRELDataFetcher:

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://developer.nrel.gov/api/nsrdb/v2/solar"
    
    def fetch_data(self, 
                   latitude: float,
                   longitude: float,
                   year: int,
                   attributes: str = "ghi,dni,dhi,air_temperature,wind_speed",
                   interval: int = 60) -> Optional[pd.DataFrame]:

        params = {
            'api_key': self.api_key,
            'wkt': f'POINT({longitude} {latitude})',
            'names': str(year),
            'attributes': attributes,
            'interval': interval,
            'utc': 'false'
        }
        
        print(f"[NREL] Fetching data for ({latitude}, {longitude}) year {year}...")
        
        try:
            response = requests.get(f"{self.base_url}/psm3-download.csv", 
                                  params=params, timeout=30)
            response.raise_for_status()
            
            from io import StringIO
            data = pd.read_csv(StringIO(response.text), skiprows=2)
            
            print(f"[NREL] ✓ Successfully fetched {len(data)} records")
            return data
            
        except Exception as e:
            print(f"[NREL] ✗ Error: {e}")
            return None
    
    def load_from_csv(self, filepath: str) -> Optional[pd.DataFrame]:
        """Load pre-downloaded NREL CSV file."""
        try:
            data = pd.read_csv(filepath, skiprows=2)
            print(f"[NREL] ✓ Loaded {len(data)} records from {filepath}")
            return data
        except Exception as e:
            print(f"[NREL] ✗ Error loading CSV: {e}")
            return None
