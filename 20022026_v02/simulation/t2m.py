import pandas as pd
import matplotlib.pyplot as plt

#prepare data
df=pd.read_csv("data/site_weather.csv")
df["date"]=pd.to_datetime(df["date"], format="mixed", dayfirst=True)

df=df.set_index("date")
daily_temp_air=df["temp_air"].resample("D").mean()
daily_temp_air=daily_temp_air.to_frame().reset_index()
#print(daily_temp_air)

daily_temp_air["date"]=pd.to_datetime(daily_temp_air["date"])

#filter by season
#1. winter
winter_data=daily_temp_air[daily_temp_air["date"].dt.month.isin([12,1,2])]
print(winter_data.head())

plt.figure(figsize=(10,5))
plt.plot(winter_data["date"], winter_data["temp_air"])
plt.title("Average Temperature @ 2m per Day (Winter)")
plt.xlabel("Date")
plt.ylabel("Temperature @ 2m")
plt.grid(False)
plt.show()

#2. spring
spring_data=daily_temp_air[daily_temp_air["date"].dt.month.isin([3,4,5])]
print(spring_data.head())

plt.figure(figsize=(10,5))
plt.plot(spring_data["date"], spring_data["temp_air"])
plt.title("Average Temperature @ 2m per Day (Spring)")
plt.xlabel("Date")
plt.ylabel("Temperature @ 2m")
plt.grid(False)
plt.show()

#3. summer
summer_data=daily_temp_air[daily_temp_air["date"].dt.month.isin([6,7,8])]
print(summer_data.head())

plt.figure(figsize=(10,5))
plt.plot(summer_data["date"], summer_data["temp_air"])
plt.title("Average Temperature @ 2m per Day (Summer)")
plt.xlabel("Date")
plt.ylabel("Temperature @ 2m")
plt.grid(False)
plt.show()

#4. autumn
autumn_data=daily_temp_air[daily_temp_air["date"].dt.month.isin([9,10,11])]
print(autumn_data.head())

plt.figure(figsize=(10,5))
plt.plot(autumn_data["date"], autumn_data["temp_air"])
plt.title("Average Temperature @ 2m per Day (Autumn)")
plt.xlabel("Date")
plt.ylabel("Temperature @ 2m")
plt.grid(False)
plt.show()