# -*- coding: utf-8 -*-
"""
Created on Fri Mar 10 10:10:17 2023

@author: baongoc.thai
"""

import os
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import datetime
import numpy as np
import math
from windrose import WindroseAxes
from scipy import interpolate
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
import statistics
from scipy.stats import pearsonr

#%% Function to import data
def ReadData(location,parameter,directory,folder):
    pd1 = pd.read_csv(directory/folder/r'{}.csv'.format(parameter+'-'+location),encoding= 'unicode_escape')
    pd1.index = pd1.pop('date and time')
    pd1.index = pd.to_datetime(pd1.index,format = "%Y-%m-%d %H:%M:%S")
    pd1 = pd1.replace(r'^\s*$', np.nan, regex=True) # replace blanks with nan
    return pd1

#%% Function to interpolate values with depth
def temperature_interpolate(dataframe, depth_model, depth_obs):
    model_raw = dataframe
    model_raw.columns = depth_model
    obslevel= depth_obs
    model = pd.DataFrame()
    
    for m in range(len(model_raw)):
        model_raw_each = model_raw.iloc[[m]]
        model_row_each_1 = model_raw_each.dropna(axis=1)
        x = np.array(model_row_each_1.columns)
        y = np.array(model_row_each_1.iloc[0])
        f = interpolate.interp1d(x,y,kind='linear',fill_value='extrapolate')
        model_each = f(obslevel)
        model = model.append(pd.Series(model_each),ignore_index=True)  
    
    model.index = dataframe.index
    model.columns = depth_obs
    return model
#%% Function to process mat file (map result from D3D-FLOW) for Heat Flux terms - integrate for whole reservoir
def ProcessMatFile(filename, grid_surface_area):
    data = read_mat(filename)['data']['Val']
    data_whole = data*grid_surface_area
    TotalArea = np.nansum(grid_surface_area)
    data_whole_ts = np.nansum(np.nansum(data_whole, axis=1), axis=1)/TotalArea
    return data_whole_ts
#%% Main block: import data
directory = Path(r'C:\Users\baongoc.thai\OneDrive - Hydroinformatics Institute Pte Ltd\Desktop\Work\5. Pandan HD\FinalSpatialPAN_WOPV_0.0015')
os.chdir(directory)

location = ['A2','F2','G1','MO1','MO2']
parameter = ['water level','temperature','horizontal velocity','vertical velocity']

no_aeration = r'Output_NoAeration\Results'
aeration = r'Output_Aeration\Results'

#Initialize dictionary
No_Aeration = dict.fromkeys(parameter, None)
Aeration = dict.fromkeys(parameter, None)

# No Aeration data
for para in parameter:
    df_temp = []
    for loc in location:
        df = ReadData(loc,para,directory,no_aeration)
        df_temp.append(df)
    No_Aeration[para] = df_temp

# Aeration data
for para in parameter:
    df_temp = []
    for loc in location:
        df = ReadData(loc,para,directory,aeration)
        df_temp.append(df)
    Aeration[para] = df_temp

#%% Main block: Plot Water Level at 5 locations
mRL = 104.66

for i in range(len(location)):
    df_aeration = Aeration['water level'][i] + mRL
    df_noaeration = No_Aeration['water level'][i] + mRL
    
    plt.plot(df_aeration.index, df_aeration.iloc[:,0], color='b', linewidth=1.5, label='With Aeration')
    plt.plot(df_noaeration.index, df_noaeration.iloc[:,0], color='m', linestyle = 'dashed', linewidth=3, label='Without Aeration')    
        
    plt.ylabel("Reservoir level (mRL)")
    plt.xlim([datetime.datetime(2019, 1, 1), datetime.datetime(2020, 1, 1)])
    plt.legend(loc='upper right')
    plt.rcParams.update({'font.size': 15})
    figure = plt.gcf()
    figure.set_size_inches(16,3.5)
    plt.title('Reservoir level at ' + location[i])
    plt.savefig('WaterLevel\\'+location[i]+"_WaterLevel.png", bbox_inches='tight',dpi=600)
    plt.close()
    print (location[i])
    
#%% Main block: Plot horizontal velocity, x & y component separately
for i in range(len(location)):
    df_aeration = Aeration['horizontal velocity'][i]
    df_noaeration = No_Aeration['horizontal velocity'][i]
    df_aeration_movingavg = df_aeration.rolling(24).mean()
    df_noaeration_movingavg = df_noaeration.rolling(24).mean()
    
    for col in df_aeration.columns.tolist():
        if (df_noaeration[col].isna().all() and df_aeration[col].isna().all()):
            continue  # to skip a value in for loop
        else:
            #Hourly output
            plt.plot(df_aeration.index, df_aeration[col], color='b', linewidth=1.5, label='With Aeration')   
            plt.plot(df_noaeration.index, df_noaeration[col], color='m', linestyle = 'dashed', linewidth=1.5, label='Without Aeration')    
                        
            plt.ylabel("Velocity (m/s)")
            plt.xlim([datetime.datetime(2019, 1, 1), datetime.datetime(2020, 1, 1)])
            plt.legend(loc='upper right')
            plt.rcParams.update({'font.size': 15})
            figure = plt.gcf()
            figure.set_size_inches(16,3.5)
            plt.title(col)
            plt.savefig('HorizontalVelocity\\'+location[i]+"_"+col.split('-')[-1]+"_"+col.split('-')[0][0]+".png", bbox_inches='tight',dpi=600)
            plt.close()
            
            #24hr moving average
            plt.plot(df_aeration_movingavg.index, df_aeration_movingavg[col], color='b', linewidth=1.5, label='With Aeration (24hr moving avg')   
            plt.plot(df_noaeration_movingavg.index, df_noaeration_movingavg[col], color='m', linestyle = 'dashed', linewidth=1.5, label='Without Aeration (24hr moving avg)')    
                        
            plt.ylabel("Velocity (m/s)")
            plt.xlim([datetime.datetime(2019, 1, 1), datetime.datetime(2020, 1, 1)])
            plt.legend(loc='upper right')
            plt.rcParams.update({'font.size': 15})
            plt.legend(fontsize=10)
            figure = plt.gcf()
            figure.set_size_inches(16,3.5)
            plt.title(col)
            plt.savefig('HorizontalVelocity\\MovingAverage'+location[i]+"_"+col.split('-')[-1]+"_"+col.split('-')[0][0]+".png", bbox_inches='tight',dpi=600)
            plt.close()
            print (col)
#%% Main block: Plot horizontal velocity, magnitude
for i in range(len(location)):
    df_aeration = Aeration['horizontal velocity'][i]
    df_noaeration = No_Aeration['horizontal velocity'][i]
    col = df_aeration.columns.tolist()
    j = 0
    while j < len(col):
        if (df_noaeration[col].isna().all() and df_aeration[col].isna().all()):
            j = j+2  # to skip a value in for loop
        else:
            print (j)
            df_aeration_temp = df_aeration.iloc[:,j:j+2]
            df_noaeration_temp = df_noaeration.iloc[:,j:j+2]
            #Caluclate magnitude of velocity
            df_aeration_temp['Magnitude'] = (df_aeration_temp.iloc[:,0]**2 + df_aeration_temp.iloc[:,1]**2) ** 0.5 
            df_noaeration_temp['Magnitude'] = (df_noaeration_temp.iloc[:,0]**2 + df_noaeration_temp.iloc[:,1]**2) ** 0.5        
            
# =============================================================================
#             #Caluclate direction
#             df_aeration_temp['Direction'] = (df_aeration_temp.iloc[:,1]/df_aeration_temp.iloc[:,2])  # Calculate cos of angle
#             df_aeration_temp['Direction'] = [math.degrees(math.acos(value)) for value in df_aeration_temp['Direction']] # Calculate acos & convert to degrees
#             
#             ax = WindroseAxes.from_ax()
#             ax.bar(df_aeration_temp['Direction'], df_aeration_temp['Magnitude'],bins=np.arange(0, 0.1, 0.01))
#             ax.set_legend()
#             plt.show()
#             plt.close()
#             
#             #Caluclate direction
#             df_noaeration_temp['Direction'] = (df_noaeration_temp.iloc[:,1]/df_noaeration_temp.iloc[:,2])  # Calculate cos of angle
#             df_noaeration_temp['Direction'] = [math.degrees(math.acos(value)) for value in df_noaeration_temp['Direction']] # Calculate acos & convert to degrees
#             
#             ax = WindroseAxes.from_ax()
#             ax.bar(df_noaeration_temp['Direction'], df_noaeration_temp['Magnitude'], normed=True, opening=0.8, edgecolor='white')
#             ax.set_legend()
#             plt.show()
#             plt.close()
# =============================================================================
            
            plt.plot(df_aeration_temp.index, df_aeration_temp['Magnitude'], color='b', linewidth=1.5, label='With Aeration')   
            plt.plot(df_noaeration_temp.index, df_noaeration_temp['Magnitude'], color='m', linestyle = 'dashed', linewidth=1.5, label='Without Aeration')    
                        
            plt.ylabel("Velocity (m/s)")
            plt.xlim([datetime.datetime(2019, 1, 1), datetime.datetime(2020, 1, 1)])
            plt.legend(loc='upper right')
            plt.rcParams.update({'font.size': 15})
            figure = plt.gcf()
            figure.set_size_inches(16,3.5)
            plt.title("Velocity at " + location[i] + " -" + col[j].split('-')[-1])
            plt.savefig('HorizontalVelocity\\'+location[i] + " -" + col[j].split('-')[-1]+"_Magnitude.png", bbox_inches='tight',dpi=600)
            plt.close()
            print (col[j])
            j = j+2
            
#%% Main block: Plot vertical velocity, magnitude
for i in range(len(location)):
    df_aeration = Aeration['vertical velocity'][i]
    df_noaeration = No_Aeration['vertical velocity'][i]
    
    df_aeration_distance = df_aeration*60*60
    df_noaeration_distance = df_noaeration*60*60
    
    for col in df_aeration.columns.tolist():
        if (df_noaeration[col].isna().all() and df_aeration[col].isna().all()):
            continue  # to skip a value in for loop
        else:
            # Plot vertical velocity
            plt.plot(df_aeration.index, df_aeration[col], color='b', linewidth=1.5, label='With Aeration')   
            plt.plot(df_noaeration.index, df_noaeration[col], color='m', linestyle = 'dashed', linewidth=1.5, label='Without Aeration')    
            plt.ylabel("Vertical velocity (m/s)")
            plt.ticklabel_format(axis = 'y', style = 'plain', useOffset = False)
            plt.xlim([datetime.datetime(2019, 1, 1), datetime.datetime(2020, 1, 1)])
            plt.legend(loc='upper right')
            plt.rcParams.update({'font.size': 15})
            figure = plt.gcf()
            figure.set_size_inches(16,3.5)
            plt.title(col)
            plt.savefig('VerticalVelocity\\'+location[i]+"_"+col.split('-')[-1]+".png", bbox_inches='tight',dpi=600)
            plt.close()
            
            #Plot vertical distance
            plt.plot(df_aeration_distance.index, df_aeration_distance[col], color='b', linewidth=1.5, label='With Aeration')   
            plt.plot(df_noaeration_distance.index, df_noaeration_distance[col], color='m', linestyle = 'dashed', linewidth=1.5, label='Without Aeration')    
            plt.ylabel("Vertical distance (m)")
            plt.ticklabel_format(axis = 'y', style = 'plain', useOffset = False)
            plt.xlim([datetime.datetime(2019, 1, 1), datetime.datetime(2020, 1, 1)])
            plt.legend(loc='upper right')
            plt.rcParams.update({'font.size': 15})
            figure = plt.gcf()
            figure.set_size_inches(16,3.5)
            plt.title('Distance t'+location[i]+"_"+col.split('-')[-1])
            plt.savefig('VerticalVelocity\\'+'Distance_'+location[i]+"_"+col.split('-')[-1]+".png", bbox_inches='tight',dpi=600)
            plt.close()
            print (col)
            
#%% Main block: Plot temperature at various depths at 5 locations
for i in range(len(location)):
    df_aeration = Aeration['temperature'][i]
    df_noaeration = No_Aeration['temperature'][i]
    df_aeration_movingavg = df_aeration.rolling(24).mean()
    df_noaeration_movingavg = df_noaeration.rolling(24).mean()
    
    for col in df_aeration.columns.tolist():
        if (df_noaeration[col].isna().all() and df_aeration[col].isna().all()):
            continue  # to skip a value in for loop
        else:
            #Hourly outputs
            plt.plot(df_aeration.index, df_aeration[col], color='b', linewidth=1.5, label='With Aeration')   
            plt.plot(df_noaeration.index, df_noaeration[col], color='m', linestyle = 'dashed', linewidth=1.5, label='Without Aeration')    
                        
            plt.ylabel("Temperature (oC)")
            plt.xlim([datetime.datetime(2019, 1, 1), datetime.datetime(2020, 1, 1)])
            plt.legend(loc='upper right')
            plt.rcParams.update({'font.size': 15})
            figure = plt.gcf()
            figure.set_size_inches(16,3.5)
            plt.title(col)
            plt.savefig('Temperature\\'+location[i]+"_"+col.split('-')[-1]+".png", bbox_inches='tight',dpi=600)
            plt.close()
            
            #24hr moving average outputs
            plt.plot(df_aeration_movingavg.index, df_aeration_movingavg[col], color='b', linewidth=1.5, label='With Aeration (24hr moving avg)')   
            plt.plot(df_noaeration_movingavg.index, df_noaeration_movingavg[col], color='m', linestyle = 'dashed', linewidth=1.5, label='Without Aeration (24hr moving avg)')    
                        
            plt.ylabel("Temperature (oC)")
            plt.xlim([datetime.datetime(2019, 1, 1), datetime.datetime(2020, 1, 1)])
            plt.legend(loc='upper right')
            plt.rcParams.update({'font.size': 15})
            figure = plt.gcf()
            figure.set_size_inches(16,3.5)
            plt.title(col)
            plt.savefig('Temperature\\MovingAverage'+location[i]+"_"+col.split('-')[-1]+".png", bbox_inches='tight',dpi=600)
            plt.close()
            print (col)
            
#%% Main block: Plot thermal stratification at 5 locations
for i in range(len(location)):
    df_aeration = Aeration['temperature'][i]
    df_noaeration = No_Aeration['temperature'][i]
    df_aeration_movingavg = df_aeration.rolling(24).mean()
    df_noaeration_movingavg = df_noaeration.rolling(24).mean()
    
    for col in df_aeration.columns.tolist():
        if (df_noaeration[col].isna().all() and df_aeration[col].isna().all()):
            continue  # to skip a value in for loop
        else:
            #Thermal stratification = delta T = T top - T bottom
            thermal_strat_noAeration = -df_noaeration[col] + df_noaeration.iloc[:,-1]
            plt.plot(thermal_strat_noAeration.index, thermal_strat_noAeration, color='m', linestyle = 'dashed', linewidth=1.5, label='Without Aeration')
            plt.ylabel("Delta T (oC)")
            plt.xlim([datetime.datetime(2019, 1, 1), datetime.datetime(2020, 1, 1)])
            plt.ylim(-0.3,2.5)
            plt.rcParams.update({'font.size': 15})
            figure = plt.gcf()
            figure.set_size_inches(16,3.5)
            plt.title("Thermal stratification at "+ location[i])
            plt.savefig('Temperature\\ThermalStrat_NoAeration'+location[i]+".png", bbox_inches='tight',dpi=600)
            plt.close()
            
            thermal_strat_Aeration = -df_aeration[col] + df_aeration.iloc[:,-1]
            plt.plot(thermal_strat_Aeration.index, thermal_strat_Aeration, color='b', linewidth=1.5, label='With Aeration')
            plt.ylabel("Delta T (oC)")
            plt.xlim([datetime.datetime(2019, 1, 1), datetime.datetime(2020, 1, 1)])
            plt.ylim(-0.3,2.5)
            plt.rcParams.update({'font.size': 15})
            figure = plt.gcf()
            figure.set_size_inches(16,3.5)
            plt.title("Thermal stratification at "+ location[i])
            plt.savefig('Temperature\\ThermalStrat_WithAeration'+location[i]+".png", bbox_inches='tight',dpi=600)
            plt.close()
            break
#%% Main block: Plot temperature contour
water_layer = [-10, -7.5, -5, -4.17, -3.33, -2.5, -1.67, -0.83]
for i in range(len(location)):
    df_noaeration = No_Aeration['temperature'][i]
    df_noaeration.columns = water_layer
    df_noaeration = df_noaeration.dropna(axis=1, how='all')
    
    df_aeration = Aeration['temperature'][i]
    df_aeration.columns = water_layer
    df_aeration = df_aeration.dropna(axis=1, how='all')
    
    df_temperature_diff = df_aeration - df_noaeration
    
    #Plot temperature difference
    x = df_temperature_diff.index
    y = df_temperature_diff.columns
    px_values = df_temperature_diff.transpose()
    fig, ax = plt.subplots()
    plt.rcParams['font.size'] = 12
    plt.rcParams['legend.fontsize'] = 'large'
    plt.rcParams['figure.titlesize'] = 'large'
    plt.rcParams["figure.figsize"] = (18, 6)
    CS = ax.contourf(x, y, px_values, cmap ='rainbow')
    fig.colorbar(CS)
    plt.ylabel('Water level wrt surface (m)')
    plt.title('Temperature difference due to aeration at ' + location[i] + "\n([With aeration] - [Without aeration])")
    plt.savefig('Temperature\\'+'Temperature difference due to aeration at ' + location[i] + '.png', bbox_inches='tight',dpi=600)
    plt.close()
    
    #Plot with aeration
    x = df_aeration.index
    y = df_aeration.columns
    px_values = df_aeration.transpose()
    fig, ax = plt.subplots()
    plt.rcParams['font.size'] = 12
    plt.rcParams['legend.fontsize'] = 'large'
    plt.rcParams['figure.titlesize'] = 'large'
    plt.rcParams["figure.figsize"] = (18, 6)
    CS = ax.contourf(x, y, px_values, cmap ='rainbow')
    fig.colorbar(CS)
    plt.ylabel('Water level wrt surface (m)')
    plt.title('[With aeration] Temperature profile at ' + location[i])
    plt.savefig('Temperature\\'+'[With aeration] Temperature profile at ' + location[i] + '.png', bbox_inches='tight',dpi=600)
    plt.close()
    
    #Plot without aeration
    x = df_noaeration.index
    y = df_noaeration.columns
    px_values = df_noaeration.transpose()
    fig, ax = plt.subplots()
    plt.rcParams['font.size'] = 12
    plt.rcParams['legend.fontsize'] = 'large'
    plt.rcParams['figure.titlesize'] = 'large'
    plt.rcParams["figure.figsize"] = (18, 6)
    CS = ax.contourf(x, y, px_values, cmap ='rainbow')
    fig.colorbar(CS)
    plt.ylabel('Water level wrt surface (m)')
    plt.title('[Without aeration] Temperature profile at ' + location[i])
    plt.savefig('Temperature\\'+'[Without aeration] Temperature profile at ' + location[i] + '.png', bbox_inches='tight',dpi=600)
    plt.close()
    print (location[i])

#%% Main block: Interpolate modelled temperature at specified depth (same as in profiler data)
# For Pandan, profiler SE505 is at A2
depth_model = np.array([10, 7.5, 5, 4.17, 3.33, 2.5, 1.67, 0.83])
depth_obs = np.array([0.5, 1, 1.5, 2, 2.5, 3, 3.5])

#Import temperature profile
temperature_profiler = pd.read_csv('SE505_A2_Pandan_TemperatureProfiler_2019.csv')
temperature_profiler.index = temperature_profiler.pop('date')
temperature_profiler.index = pd.to_datetime(temperature_profiler.index, format = '%d/%m/%Y %H:%M')
temperature_profiler = temperature_profiler.sort_index()
temperature_profiler.columns = depth_obs

# Interpolate by time for temperature
temperature_profiler_min = temperature_profiler.resample('1Min').mean()
temperature_profiler_int = temperature_profiler_min.interpolate('time', limit = 120)
temperature_profiler_hourly = temperature_profiler_int.resample('1H').asfreq()
temperature_profiler_hourly_movingavg = temperature_profiler_hourly.rolling(24).mean()

# Plot for modelled & observed temperature at each depth - for without aeration
df_noaeration = No_Aeration['temperature'][0] #A2 is index 0
noaeration_temp_inter = temperature_interpolate(df_noaeration, depth_model, depth_obs)
noaeration_temp_inter_movingavg = noaeration_temp_inter.rolling(24).mean()

for n in range(len(depth_obs)):
    #Hourly
    plt.plot(noaeration_temp_inter.index, noaeration_temp_inter[depth_obs[n]], color='green', linewidth=1.5,label="Model")
    plt.scatter(temperature_profiler_hourly.index,temperature_profiler_hourly[depth_obs[n]], s=1,  color='black',label="Observation")
    plt.xlim([datetime.datetime(2019, 1, 1), datetime.datetime(2020, 1, 1)])
    plt.ylim([25,35])
    plt.ylabel("Temperature (degree C)")
    plt.title("Temperature at " + location[0] + ", "+ str(depth_obs[n])+"m depth (Without Aeration)")
    plt.legend(loc='upper right')
    plt.rcParams.update({'font.size': 15})
    figure = plt.gcf() # get current figure
    figure.set_size_inches(16, 4)
    plt.savefig('Temperature\\'+"WithoutAeration_Temperature_" + location[0]+"_"+str(depth_obs[n])+"m.png", bbox_inches='tight',dpi=600)
    plt.close()
    
    #Moving avg
    plt.plot(noaeration_temp_inter_movingavg.index, noaeration_temp_inter_movingavg[depth_obs[n]], color='green', linewidth=1.5,label="Model (24hr moving avg)")
    plt.scatter(temperature_profiler_hourly_movingavg.index,temperature_profiler_hourly_movingavg[depth_obs[n]], s=1,  color='black',label="Observation (24hr moving avg)")
    plt.xlim([datetime.datetime(2019, 1, 1), datetime.datetime(2020, 1, 1)])
    plt.ylim([25,35])
    plt.ylabel("Temperature (degree C)")
    plt.title("Temperature at " + location[0] + ", "+ str(depth_obs[n])+"m depth (Without Aeration)")
    plt.legend(loc='upper right')
    plt.rcParams.update({'font.size': 15})
    plt.legend(fontsize=10)
    figure = plt.gcf() # get current figure
    figure.set_size_inches(16, 4)
    plt.savefig('Temperature\\MovingAverage_'+"WithoutAeration_Temperature_" + location[0]+"_"+str(depth_obs[n])+"m.png", bbox_inches='tight',dpi=600)
    plt.close()
    print (depth_obs[n])
    
# Plot for modelled & observed temperature at each depth - for with aeration
df_aeration = Aeration['temperature'][0] #A2 is index 0
aeration_temp_inter = temperature_interpolate(df_aeration, depth_model, depth_obs)
aeration_temp_inter_movingavg = aeration_temp_inter.rolling(24).mean()

for n in range(len(depth_obs)):
    #Hourly
    plt.plot(aeration_temp_inter.index, aeration_temp_inter[depth_obs[n]], color='green', linewidth=1.5,label="Model")
    plt.scatter(temperature_profiler_hourly.index,temperature_profiler_hourly[depth_obs[n]], s=1,  color='black',label="Observation")
    plt.xlim([datetime.datetime(2019, 1, 1), datetime.datetime(2020, 1, 1)])
    plt.ylim([25,35])
    plt.ylabel("Temperature (degree C)")
    plt.title("Temperature at " + location[0] + ", "+ str(depth_obs[n])+"m depth (With Aeration)")
    plt.legend(loc='upper right')
    plt.rcParams.update({'font.size': 15})
    figure = plt.gcf() # get current figure
    figure.set_size_inches(16, 4)
    plt.savefig('Temperature\\'+"WithAeration_Temperature_" + location[0]+"_"+str(depth_obs[n])+"m.png", bbox_inches='tight',dpi=600)
    plt.close()
    
    #Moving avg
    plt.plot(aeration_temp_inter_movingavg.index, aeration_temp_inter_movingavg[depth_obs[n]], color='green', linewidth=1.5,label="Model (24hr moving avg)")
    plt.scatter(temperature_profiler_hourly_movingavg.index,temperature_profiler_hourly_movingavg[depth_obs[n]], s=1, color='black',label="Observation (24hr moving avg)")
    plt.xlim([datetime.datetime(2019, 1, 1), datetime.datetime(2020, 1, 1)])
    plt.ylim([25,35])
    plt.ylabel("Temperature (degree C)")
    plt.title("Temperature at " + location[0] + ", "+ str(depth_obs[n])+"m depth (With Aeration)")
    plt.legend(loc='upper right')
    plt.rcParams.update({'font.size': 15})
    plt.legend(fontsize=10)
    figure = plt.gcf() # get current figure
    figure.set_size_inches(16, 4)
    plt.savefig('Temperature\\MovingAverage_'+"WithAeration_Temperature_" + location[0]+"_"+str(depth_obs[n])+"m.png", bbox_inches='tight',dpi=600)
    plt.close()
    print (depth_obs[n])

# Compare profiler and modelled results
temperature_profiler_hourly_comparison = temperature_profiler_hourly.loc[aeration_temp_inter.index]
temperature_profiler_hourly_comparison = temperature_profiler_hourly_comparison.dropna()
aeration_temp_inter = aeration_temp_inter.loc[temperature_profiler_hourly_comparison.index]
noaeration_temp_inter = noaeration_temp_inter.loc[aeration_temp_inter.index]

#Compare with aeration results
difference_aeration = abs(aeration_temp_inter - temperature_profiler_hourly_comparison)
mae_aeration = difference_aeration.mean(axis = 0, skipna = True)
mse_aeration = (difference_aeration**2).mean(axis = 0, skipna = True)
rmse_aeration = (difference_aeration**2).mean(axis = 0, skipna = True)**0.5
pearson_aeration = pd.Series([pearsonr(aeration_temp_inter[col],temperature_profiler_hourly_comparison[col])[0] for col in depth_obs])
pearson_aeration.index = rmse_aeration.index
std_modelled_aeration = aeration_temp_inter.std()
mean_modelled_aeration = aeration_temp_inter.mean()
std_temperature_profiler_hourly_comparison = temperature_profiler_hourly_comparison.std()
mean_temperature_profiler_hourly_comparison = temperature_profiler_hourly_comparison.mean()
aeration_stats_results = pd.concat([mae_aeration, mse_aeration, rmse_aeration, pearson_aeration, std_modelled_aeration, mean_modelled_aeration, std_temperature_profiler_hourly_comparison, mean_temperature_profiler_hourly_comparison], axis=1)
aeration_stats_results.columns = ['mae','mse','rmse','pearson r','modelled std','modelled mean','obs std','obs mean']
aeration_stats_results.to_csv('Vertical temperature profile stats at A2 (with aeration).csv')

#Compare with aeration results
difference_noaeration = abs(noaeration_temp_inter - temperature_profiler_hourly_comparison)
mae_noaeration = difference_noaeration.mean(axis = 0, skipna = True)
mse_noaeration = (difference_noaeration**2).mean(axis = 0, skipna = True)
rmse_noaeration = (difference_noaeration**2).mean(axis = 0, skipna = True)**0.5
pearson_noaeration = pd.Series([pearsonr(noaeration_temp_inter[col],temperature_profiler_hourly_comparison[col])[0] for col in depth_obs])
pearson_noaeration.index = rmse_noaeration.index
std_modelled_noaeration = noaeration_temp_inter.std()
mean_modelled_noaeration = noaeration_temp_inter.mean()
noaeration_stats_results = pd.concat([mae_noaeration, mse_noaeration, rmse_noaeration, pearson_noaeration, std_modelled_noaeration, mean_modelled_noaeration, std_temperature_profiler_hourly_comparison, mean_temperature_profiler_hourly_comparison], axis=1)
noaeration_stats_results.columns = ['mae','mse','rmse','pearson r','modelled std','modelled mean','obs std','obs mean']
noaeration_stats_results.to_csv('Vertical temperature profile stats at A2 (without aeration).csv')

#%% Main block: Read mat file
from pymatreader import read_mat
# Extract surface grid cell areas from mat file
SurfaceArea = read_mat('grid cell surface area.mat')['data']['Val']
Grid_x =  read_mat('grid cell surface area.mat')['data']['X']
Grid_y =  read_mat('grid cell surface area.mat')['data']['Y']
TotalArea = np.nansum(SurfaceArea)
SurfaceArea_1 = SurfaceArea[1:,1:]

# With Aeration
import glob
files = glob.glob("*WithAeration*.mat")
column_names = [file.split('_')[0] for file in files]
WholeReservoir_Aeration = []

for file in files:
    data = ProcessMatFile(file, SurfaceArea_1)
    WholeReservoir_Aeration.append(data)

WholeReservoir_Aeration_df = pd.DataFrame(WholeReservoir_Aeration).transpose()
WholeReservoir_Aeration_df.columns = column_names
WholeReservoir_Aeration_df.index = df.index
WholeReservoir_Aeration_df.to_csv('WholeReservoir_HeatFlux_WithAeration.csv')

# Without Aeration
import glob
files = glob.glob("*NoAeration*.mat")
column_names = [file.split('_')[0] for file in files]
WholeReservoir_NoAeration = []

for file in files:
    data = ProcessMatFile(file, SurfaceArea_1)
    WholeReservoir_NoAeration.append(data)

WholeReservoir_NoAeration_df = pd.DataFrame(WholeReservoir_NoAeration).transpose()
WholeReservoir_NoAeration_df.columns = column_names
WholeReservoir_NoAeration_df.index = df.index
WholeReservoir_NoAeration_df.to_csv('WholeReservoir_HeatFlux_NoAeration.csv')


