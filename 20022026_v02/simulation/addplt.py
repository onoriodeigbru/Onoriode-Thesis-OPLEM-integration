#---- import modules ----
from c.csv_loader import load_weather_csv
from models.simulate_generation import *
from plots.pv import *
from plots.wind import *

#---- loading the csv data of the weather downloaded from https://www.renewables.ninja/ ----
weather = load_weather_csv("data/site_weather.csv")

#---- to check if the weather data was accurately loaded ----
print (weather.head())

#---- to simulate the data ----
results = simulate_system(weather)

#---- printing outcome for confirmation ----
print(results.head())

results["time"] = pd.to_datetime(results["time"], format="%H:%M:%S", errors="coerce")
results["time"] = pd.date_range(start="2021-01-01", periods=len(results), freq="H")

def plot_seasonal_power(results, power_key, plot_func):
    import pandas as pd

    # Ensure proper datetime (FIX if only time exists)
    if results["time"].dt.date.nunique() == 1:
        results["time"] = pd.date_range(start="2021-01-01",
                                        periods=len(results),
                                        freq="H")  # adjust if needed

    # Define seasons
    def get_season(month):
        if month in [12, 1, 2]:
            return "winter"
        elif month in [3, 4, 5]:
            return "spring"
        elif month in [6, 7, 8]:
            return "summer"
        else:
            return "autumn"

    results["season"] = results["time"].dt.month.apply(get_season)

    power = results[power_key].values

    # Loop through seasons and call your existing plot function
    for season in ["winter", "spring", "summer", "autumn"]:
        idx = results["season"] == season
        print(f"Plotting {power_key} for {season}...")
        plot_func(power[idx])

plot_seasonal_power(results, "pv_power", plot_pv_power)
plot_seasonal_power(results, "wind_power", plot_wind_power)
