#import modules
import os
from os.path import normpath, join
import copy
import pandas as pd
import numpy as np
import pickle
import time
import matplotlib.pyplot as plt

from oplem.Network_3ph_pf import Network_3ph
import oplem.Assets as AS
from oplem.Participant import Participant
from oplem.Market import ToU_market

from c.csv_loader import load_weather_csv
from models.simulate_generation import simulate_system

np.random.seed(1000)
#######################################
### RUN OPTIONS
#######################################

dt_raw = 1/60
T_raw = int(24/dt_raw) #Number of data time intervals
dt = 60/60 #5 minute time intervals
T = int(24/dt) #Number of intervals
dt_ems = 60/60
T_ems = int(24/dt_ems)

path_string = normpath('Results\\Central\\')
if not os.path.isdir(path_string):
    os.makedirs(path_string)

#---- loading the csv data of the weather downloaded from https://www.renewables.ninja/ ----
weather = load_weather_csv("data/site_weather.csv")
#---- to check if the weather data was accurately loaded ----
#print (weather.head())
#---- to simulate the data ----
results = simulate_system(weather)
print (results.head())
results["time"] = pd.to_datetime(results["time"], format="%H:%M:%S", errors="coerce")
pv = results["total_power"].values 
Ta = results["temp_air"].values

#######################################
### STEP 0: Load Data
#######################################
#### 1)wholesale data
prices_path = os.path.join("data", "half-hourly-wholesale-prices-MWh-29-06-2021.csv")
prices_wsm = pd.read_csv(prices_path, delimiter='\t').values.astype(float)
#prices_wsm = np.array(prices_wsm)  #prices £/MWh
dt_wsm = 24/len(prices_wsm)
T_wsm = len(prices_wsm)
prices_wsm_ems = np.zeros((T_ems,2)) # col0 for day ahead, col1 for intraday
if dt_ems <= dt_wsm:
    for t in range(T_wsm):
        prices_wsm_ems[t*int(dt_wsm/dt_ems) : (t+1)*int(dt_wsm/dt_ems),:] = prices_wsm[t,:]/1e3
else:
    for t in range(T_ems):
        prices_wsm_ems[t,:] = np.mean(prices_wsm[t*int(dt_ems/dt_wsm) : (t+1)*int(dt_ems/dt_wsm),:], axis=0)
        prices_wsm_ems[t,:] = prices_wsm_ems[t,:]/1e3


### 2) Load Data
Loads_data_path = os.path.join("data", "Loads_1min.csv")    
Loads_raw = pd.read_csv(Loads_data_path, index_col=0).values
N_loads_raw = Loads_raw.shape[1]
Loads = Loads_raw.transpose().reshape(-1,int(dt/dt_raw)).mean(1).reshape(N_loads_raw,-1).transpose()
Load_ems = Loads.transpose().reshape(-1,int(dt_ems/dt)).mean(1).reshape(N_loads_raw,-1).transpose()

#### 3) PV Data
# --- Ensuring correct resolution ---
# For when the simulation is likely 15-min → hence the need to convert to dt_ems (1 hour)
sim_dt = (results["time"].iloc[1] - results["time"].iloc[0]).total_seconds() / 3600
pv_series = pd.Series(pv)

if sim_dt < dt_ems: #In case there is need to convert the timescale from 15-min → hourly
    factor = int(dt_ems / sim_dt)
    pv_resampled = pv_series.groupby(np.arange(len(pv_series)) // factor).mean().values
else:
    pv_resampled = pv

PVpu = pv_resampled[:T_ems] #Ensuring length correctness
pv_max = np.max(PVpu) #Normalizing the PV dataset to per-unit
if pv_max == 0:
    raise ValueError("PV data is all zeros — check simulation")

PVpu = PVpu / pv_max

### 3) Temperature
Ta = np.nan_to_num(Ta, nan=25) 
Ta_series = pd.Series(Ta)

sim_dt = (results["time"].iloc[1] - results["time"].iloc[0]).total_seconds() / 3600

if sim_dt < dt_ems:
    factor = int(dt_ems / sim_dt)
    Ta_resampled = Ta_series.groupby(np.arange(len(Ta_series)) // factor).mean().values
else:
    Ta_resampled = Ta

Ta_ems = Ta_resampled[:T_ems]

### 4) Ensuring numerical stability
PVpu = np.nan_to_num(PVpu, nan=0.0)
PVpu = np.clip(PVpu, 0, 1)

#######################################
### STEP 1: Setup the network
#######################################
network = Network_3ph() 
network.setup_network_eulv_reduced()
# set bus voltage and capacity limits
network.set_pf_limits(0.95*network.Vslack_ph, 1.05*network.Vslack_ph,
                      2000*1e3/network.Vslack_ph)
N_buses = network.N_buses
N_phases = network.N_phases

#buses that contain loads
load_buses = np.where(np.abs(network.bus_df['Pa'])+np.abs(network.bus_df['Pb'])+np.abs(network.bus_df['Pc'])>0)[0]
load_phases = []
N_load_bus_phases=0
for load_bus_idx in range(len(load_buses)):
    phase_list = []
    if np.abs(network.bus_df.iloc[load_buses[load_bus_idx]]['Pa']) > 0:
          phase_list.append(0)
    if np.abs(network.bus_df.iloc[load_buses[load_bus_idx]]['Pb']) > 0:
          phase_list.append(1)
    if np.abs(network.bus_df.iloc[load_buses[load_bus_idx]]['Pc']) > 0:
          phase_list.append(2)
    load_phases.append(np.array(phase_list))  
    N_load_bus_phases += len(phase_list)
N_loads = load_buses.size

#######################################
### STEP 2: setup parameters
######################################
#### 1) Home PV parameters
N_pv = int(np.ceil(0.6*N_loads)) #nbr of homes with PV 
pv_locs = np.random.choice(N_loads, N_pv, replace=False)# [0,3,4,6] #
Ppv_home_nom = 8#power rating of the PV generation 

### 2) Home battery parameters
N_es = int(np.ceil(0.3*N_loads)) #[1,3,5,6]
es_locs = np.random.choice(N_loads, N_es, replace=False)
Pbatt_max = 4 
Ebatt_max = 8 
c1_batt_deg = 0.005 #Battery degradation costs

### 3) building parameters
N_hp = int(np.ceil(0.3*N_loads)) 
hp_locs = np.random.choice(N_loads, N_hp, replace=False) #[2,4,5,6]
Tmax = 18 # degree celsius
Tmin = 16 # degree celsius
T0 = 17 # degree centigrade
#Parameters from 'Aggregate Flexibility of Thermostatically Controlled Loads'
heatmax = 5.6 #kW Max heat supplied
coolmax = 5.6 #kW Max cooling
CoP_heating = 2.5# coefficient of performance - heating
CoP_cooling = 2.5# coefficient of performance - cooling
C = 2 # kWh/ degree celsius
R = 2 #degree celsius/kW

#######################################
### STEP 3: setup the assets 
######################################
## We have one participant per bus, i.e., a participant is a home owner
assets_per_participant = [ [] for _ in range(N_loads) ]

#55 Homes
Loads_actual = Loads[:,:N_loads]
for i in range(N_loads):
    Pnet = Loads_actual[:,i]
    Qnet = Loads_actual[:,i]*0.05
    load_i = AS.NondispatchableAsset(Pnet, Qnet, load_buses[i], dt, T, dt_ems, T_ems, phases=load_phases[i])
    load_i.Pnet_pred = load_i.Pnet
    load_i.Qnet_pred = load_i.Qnet
    assets_per_participant[i].append(load_i)
    
    if i in pv_locs:
        Pnet_pv_i = -PVpu*Ppv_home_nom 
        pv_i = AS.CurtailableAsset(Pnet_pv_i, np.zeros(T_ems), load_buses[i], dt, T, dt_ems, T_ems, LoG='gen', phases=load_phases[i], curt=True)
        pv_i.Pnet_pred = pv_i.Pnet
        assets_per_participant[i].append(pv_i)
    
    if i in es_locs:
        Emax_i = Ebatt_max*np.ones(T_ems)
        Emin_i = np.zeros(T_ems)
        ET_i = Ebatt_max*0.5
        E0_i = Ebatt_max*0.5        
        Pmax_i = Pbatt_max*np.ones(T_ems)
        Pmin_i = -Pbatt_max*np.ones(T_ems)
        batt_i = AS.StorageAsset(Emax_i, Emin_i, Pmax_i, Pmin_i, E0_i, ET_i, load_buses[i], dt, T, dt_ems, T_ems, phases=load_phases[i], c_deg_lin = c1_batt_deg)
        assets_per_participant[i].append(batt_i)
    
    if i in hp_locs:
        bldg_i = AS.BuildingAsset(Tmax*np.ones(T_ems), Tmin*np.ones(T_ems), heatmax, coolmax, T0, C, R, CoP_heating, CoP_cooling, Ta_ems, load_buses[i], dt, T, dt_ems, T_ems, phases=load_phases[i])
        assets_per_participant[i].append(bldg_i)


##############################################
### STEP 4: Linking assets to participant object
############################################
participants = []
for i in range(N_loads):
    participant = Participant(i+1, assets_per_participant[i])
    participants.append(participant)


##############################
### STEP 5: setup the Market
############################
## 1)setup prices
TOUP = prices_wsm_ems[:,0]
TOUP = np.expand_dims(TOUP, axis=1)
TOUP = np.repeat(TOUP, network.N_buses, axis=1)

FiT = 0.06*np.ones(T_ems) 
FiT = np.expand_dims(FiT, axis=1)
FiT = np.repeat(FiT, network.N_buses, axis=1)

### 2)initialise market
ToU = ToU_market(participants, dt_ems, T_ems, TOUP, price_exp=FiT, t_ahead_0=0, network=network)
ToU.P_import = np.full(T_ems, 1e6)
ToU.P_export = np.full(T_ems, 1e6)
prices_wsm_ems = np.nan_to_num(prices_wsm_ems, nan=0.1)

def assert_valid(name, arr):
    if np.isnan(arr).any():
        raise ValueError(f"{name} contains NaN")
    if np.isinf(arr).any():
        raise ValueError(f"{name} contains Inf")

assert_valid("PVpu", PVpu)
assert_valid("Loads", Loads)
assert_valid("Temperature", Ta)
assert_valid("Prices", prices_wsm_ems)
assert_valid("P_import", ToU.P_import)

ToU.prices = prices_wsm_ems[:, 0]
ToU.loads = Load_ems[:, 0]

###3) Run market clearing
start_time = time.time() 
market_clearing_outcome, schedules = ToU.market_clearing()
elapsed = time.time() - start_time
print('ToU time=' + str(elapsed) + ' sec'  )

pickle.dump((market_clearing_outcome), open( "Results\\ToU\\mc.p", "wb" ) )
pickle.dump((schedules), open( "Results\\ToU\\schedules.p", "wb" ) )
print(market_clearing_outcome)

### Store assets states
Elist, Tinlist = [], []
for p, par in enumerate(participants):
    for a, asset in enumerate(par.assets):
        if isinstance(asset, AS.StorageAsset): #assets[par][a] == 'es':
            Elist.append(asset.E_ems)            
        elif isinstance(asset, AS.BuildingAsset): #assets[par][a] == 'hp':
            Tinlist.append(asset.Tin_ems)
pickle.dump((Elist), open( "Results\\ToU\\Ebatt.p", "wb" ) )
pickle.dump((Tinlist), open( "Results\\ToU\\Tin.p", "wb" ) )


###4) simulate network
network_pf = ToU.simulate_network_3ph()
network_pf = network_pf[:T_ems] #to take only T_ems entries
voltage = np.zeros((T_ems, 2))

for t, network_sim in enumerate(network_pf):
    buses_Vpu = np.abs(network_sim.v_net_res)/network_sim.Vslack_ph  
    voltage[t, 0] = np.min(buses_Vpu) 
    voltage[t, 1] = np.max(buses_Vpu)
    
    if np.max(buses_Vpu)>1.05:
        print('for t ={}, Vmax={} at index: {}'.format(t, np.max(buses_Vpu), np.argmax(buses_Vpu)))
    elif np.min(buses_Vpu) <0.95:
        print('for t ={}, Vmin={} at index: {}'.format(t, np.min(buses_Vpu), np.argmin(buses_Vpu) ))

pickle.dump((voltage), open( "Results\\ToU\\voltage.p", "wb" ) )

