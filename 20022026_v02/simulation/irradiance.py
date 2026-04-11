import pandas as pd
import matplotlib.pyplot as plt

#prepare data
df=pd.read_csv("data/site_weather.csv")
df["date"]=pd.to_datetime(df["date"], format="mixed", dayfirst=True)

"""
def get_season(date):
    m=date.month
    if m in [12,1,2]:
        return "Winter"
    if m in [3,4,5]:
        return "Spring"
    if m in [6,7,8]:
        return "Summer"
    else:
        return "Autumn"
"""  

#df["season"]=df["date"].apply(get_season)

#season_avg=df.groupby("season")["ghi"].mean()

#print (season_avg)

#season_avg=season_avg.reindex(["Winter","Spring","Summer","Autumn"])

#plt.bar(season_avg.index,season_avg.values)

#plt.xlabel("Season")
#plt.ylabel("Average Irradiance")
#plt.title("Average Irradiance per season")

#plt.show()

df=df.set_index("date")
daily_ghi=df["ghi"].resample("D").mean()
daily_ghi=daily_ghi.to_frame().reset_index()
#print(daily_ghi)

daily_ghi["date"]=pd.to_datetime(daily_ghi["date"])

#filter by season
#1. winter
winter_data=daily_ghi[daily_ghi["date"].dt.month.isin([12,1,2])]
print(winter_data.head())

plt.figure(figsize=(10,5))
plt.plot(winter_data["date"], winter_data["ghi"])
plt.title("Average Irradiance per Day (Winter)")
plt.xlabel("Date")
plt.ylabel("Irradiance")
plt.grid(False)
plt.show()

#2. spring
spring_data=daily_ghi[daily_ghi["date"].dt.month.isin([3,4,5])]
print(spring_data.head())

plt.figure(figsize=(10,5))
plt.plot(spring_data["date"], spring_data["ghi"])
plt.title("Average Irradiance per Day (Spring)")
plt.xlabel("Date")
plt.ylabel("Irradiance")
plt.grid(False)
plt.show()

#3. summer
summer_data=daily_ghi[daily_ghi["date"].dt.month.isin([6,7,8])]
print(summer_data.head())

plt.figure(figsize=(10,5))
plt.plot(summer_data["date"], summer_data["ghi"])
plt.title("Average Irradiance per Day (Summer)")
plt.xlabel("Date")
plt.ylabel("Irradiance")
plt.grid(False)
plt.show()

#4. autumn
autumn_data=daily_ghi[daily_ghi["date"].dt.month.isin([9,10,11])]
print(autumn_data.head())

plt.figure(figsize=(10,5))
plt.plot(autumn_data["date"], autumn_data["ghi"])
plt.title("Average Irradiance per Day (Autumn)")
plt.xlabel("Date")
plt.ylabel("Irradiance")
plt.grid(False)
plt.show()