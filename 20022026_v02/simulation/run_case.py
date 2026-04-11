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
pv = results["pv_power"].values
wind = results["wind_power"].values

winter_idx = results["season"].values == "winter"
spring_idx = results["season"].values == "spring"
summer_idx = results["season"].values == "summer"
autumn_idx = results["season"].values == "autumn"

#plot_pv_power(pv[winter_idx])
#plot_pv_power(pv[spring_idx])
plot_pv_power(pv[summer_idx])
#plot_pv_power(pv[autumn_idx])

#---- plotting results from the pv simulation ----
#plot_pv_timeseries(results["ghi"].values,
#                   results["pv_power"].values)

plot_pv_power(results["pv_power"].values)
plot_pv_power(results["pv_power"].values[3624:5832]) #summer
#---- plotting results from the wind simulation ----
#plot_wind_power_curve(results["wind_speed"].values,
 #                     results["wind_power"].values)

#plot_wind_timeseries(results["wind_speed"].values, 
 #                    results["wind_power"].values)

plot_wind_power(wind[winter_idx])
#plot_wind_power(wind[spring_idx])
#plot_wind_power(wind[summer_idx])
#plot_wind_power(wind[autumn_idx])

plot_wind_power(results["wind_power"].values)
plot_wind_power(results["wind_power"].values[3624:5832]) #summer