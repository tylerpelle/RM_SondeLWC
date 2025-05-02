#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 29 16:06:13 2025

@author: tpelle
This is a copy-cat script to re-write the freq_folder.py script that was 
originally written by Todd to try and debug it.
"""

import os
import glob
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import metpy.calc as mpcalc
from metpy.plots import add_metpy_logo, Hodograph, SkewT
from metpy.units import units

# ---------------------------
# Custom Central Difference Function (used to compute N)
# ---------------------------
def central_diff(x, y):
    """
    Compute the derivative dy/dx using central differences.
    x and y must be numpy arrays of the same shape.
    The first and last points use forward and backward differences respectively.
    """
    dydx = np.empty_like(y)
    # Forward difference for the first element
    dydx[0] = (y[1] - y[0]) / (x[1] - x[0])
    # Backward difference for the last element
    dydx[-1] = (y[-1] - y[-2]) / (x[-1] - x[-2])
    # Central differences for interior points
    for i in range(1, len(y) - 1):
        dydx[i] = (y[i + 1] - y[i - 1]) / (x[i + 1] - x[i - 1])
    return dydx

# -----------------------------
# Get csv and raw files
# -----------------------------
base_path = r'/Users/tylerpelle/Desktop/Rainmaker_Projects/SlwcAnalysis/TestData_NEW/'
csv_file = glob.glob(os.path.join(base_path, "*.csv"))
raw_file = glob.glob(os.path.join(base_path, "*.raw"))
csv_path_out = base_path + 'NewOutput_withslw.csv'
plt_path_out = base_path + 'NewOutput_SkewTFreq.png'

#Read raw data into Pandas data frame
SLW = pd.read_csv(raw_file[0], header=None, names=['Date', 'Time', 'Raw_Data'])
SLW['Date'] = SLW['Date'].astype(str).str.strip()
SLW['Time'] = SLW['Time'].astype(str).str.strip()
combined_SLW = SLW['Date'] + ' ' + SLW['Time']
combined_SLW = combined_SLW.apply(lambda x: re.sub(' +', ' ', x))
# Use explicit format for SLW timestamps (adjust if needed)
SLW['time'] = pd.to_datetime(combined_SLW, format='%Y/%m/%d %H:%M:%S.%f', errors='coerce')
SLW.drop(['Date', 'Time'], axis=1, inplace=True)

# -----------------------------
# Read sonde data into Pandas data frame
# -----------------------------
sonde = pd.read_csv(
    csv_file[3],
    skiprows=25,
    names=[
        'Date', 'Time', 'milliseconds', 'seconds_since_midnight', 'elapsed_minutes',
        'Altitude_km', 'Pressure_mb', 'Temperature_C', 'Raw_Temp', 'RH_percent',
        'iMet_frostpoint_C', 'iMet_internal_temp_C', 'iMet_battery_voltage_V',
        'iMet_theta_K', 'iMet_temp_pressure_sensor_C', 'iMet_temp_humidity_sensor_C', 
        'iMet_ascent_rate_m_s', 'iMet_water_vapor_mixing_ratio_ppmv', 'iMet_total_column_water_mm',
        'GPS_latitude', 'GPS_longitude', 'GPS_altitude_km', 'GPS_num_satellites',
        'GPS_pressure_mb', 'GPS_wind_speed_m_s', 'GPS_wind_direction_deg', 'GPS_ascent_rate_m_s', 
        'GPS_east_velocity_m_s', 'GPS_north_velocity_m_s', 'GPS_up_velocity_m_s', 
        'GPS_time_h_m_s_GMT', 'GPS_heading_from_launch_deg', 'GPS_elevation_angle_from_launch_deg',
        'GPS_distance_from_launch_km', 'predicted_landing_latitude', 'predicted_landing_longitude',
        'predicted_time_to_landing_min'
    ]
)
#Fix time
sonde['Date'] = sonde['Date'].astype(str).str.strip()
sonde['Time'] = sonde['Time'].astype(str).str.strip()
combined_sonde = sonde['Date'] + ' ' + sonde['Time']
combined_sonde = combined_sonde.apply(lambda x: re.sub(' +', ' ', x))
# Use explicit format for sonde timestamps (adjust if needed)
sonde['time'] = pd.to_datetime(combined_sonde, format='%Y-%m-%d %H:%M:%S', errors='coerce')
sonde.drop(['Date', 'Time'], axis=1, inplace=True)
# Add milliseconds for more precise time matching
sonde['pro_time'] = sonde['time'] + pd.to_timedelta(sonde['milliseconds'], unit='ms')
sonde = sonde.dropna(subset=['pro_time']).reset_index(drop=True)

# -----------------------------
# 3. Filter SLW Data by Pattern
# -----------------------------
pattern = '0103092901'
# UWCr: SLW rows where Raw_Data contains the pattern
UWCr = SLW[SLW['Raw_Data'].str.contains(pattern, na=False)].reset_index(drop=True)
# Sonde_SLW: the row immediately following a pattern row
Sonde_SLW = SLW.loc[SLW['Raw_Data'].str.contains(pattern, na=False).shift(-1, fill_value=False)].reset_index(drop=True)

# -----------------------------------------------
# 4. Merge sonde with SLW (Sonde_SLW) Data (Left Join)
# -----------------------------------------------
pro_data_matched = pd.merge_asof(
    sonde.sort_values('pro_time'),
    Sonde_SLW[['time']].sort_values('time'),
    left_on='pro_time',
    right_on='time',
    tolerance=pd.Timedelta('2s'),
    direction='nearest'
).reset_index(drop=True)

# -----------------------------------------------
# 5. Merge in the UWCr data to get 'Raw_Data'
# (Left join so that every sonde record is kept;
#  if no match, 'Raw_Data' will be NaN)
# -----------------------------------------------
merged_data = pd.merge_asof(
    pro_data_matched.sort_values('pro_time'),
    UWCr[['time', 'Raw_Data']].sort_values('time'),
    left_on='pro_time',
    right_on='time',
    tolerance=pd.Timedelta('2s'),
    direction='nearest'
).reset_index(drop=True)

# -----------------------------------------------
# 6. Extract Fields from 'Raw_Data'
# If no valid string is present, the fields will be NaN.
# -----------------------------------------------
slw0 = []
for idx, row in merged_data.iterrows():
    raw = row['Raw_Data']
    if not isinstance(raw, str) or len(raw) < 25:
        slw_fields = pd.Series([np.nan, np.nan, np.nan, np.nan, np.nan])
    try:
        ch1f = int(raw[13:17], 16) / 1000
    except Exception:
        ch1f = np.nan
    try:
        ch2f = int(raw[17:21], 16) / 1000
    except Exception:
        ch2f = np.nan
    try:
        ch1c = int(raw[21], 16)
    except Exception:
        ch1c = np.nan
    try:
        ch2c = int(raw[22], 16)
    except Exception:
        ch2c = np.nan
    try:
        bv = int(raw[23:25], 16) * 25
    except Exception:
        bv = np.nan
    slw0.append([ch1f,ch2f,ch1c,ch2c,bv])
slw_fields = pd.DataFrame(slw0)
slw_fields.columns = [
    'TWC Wire Frequency (Hz)', 'SLWC Wire Frequency (Hz)',
    'TWC Measurement Cycles', 'SLWC Measurement Cycles',
    'Battery Voltage (mV)'
]

# -----------------------------------------------
# 7. Create the Combined DataFrame
# (Use the sonde data from pro_data_matched and add the SLW fields)
# -----------------------------------------------
UWC2 = pd.DataFrame()
UWC2['time'] = merged_data['pro_time']  # use sonde time as reference
UWC2['Time2'] = UWC2['time'].dt.strftime('%H:%M:%S')
UWC2 = pd.concat([UWC2, slw_fields], axis=1)
UWC2['Altitude (km)'] = pro_data_matched['Altitude_km']
UWC2['Pressure (mb)'] = pro_data_matched['Pressure_mb']
UWC2['Temperature (C)'] = pro_data_matched['Temperature_C']
UWC2['RH (%)'] = pro_data_matched['RH_percent']
UWC2['Latitude'] = pro_data_matched['GPS_latitude']
UWC2['Longitude'] = pro_data_matched['GPS_longitude']
UWC2['Ascent Rate (m/s)'] = pro_data_matched['iMet_ascent_rate_m_s']
UWC2['GPS_wind_speed_m_s'] = pro_data_matched['GPS_wind_speed_m_s']
UWC2['GPS_wind_direction_deg'] = pro_data_matched['GPS_wind_direction_deg']
UWC2 = UWC2[UWC2['GPS_wind_speed_m_s'] < 500]
start_time = sonde.loc[sonde['iMet_ascent_rate_m_s'] > 1, 'pro_time'].min().floor('s')
end_time = sonde['pro_time'].max().floor('s')

# -----------------------------------------------
# 8. Create output dataframe to hold combined SLWC and sonde data
# (Use the sonde data from pro_data_matched and add the SLW fields)
# -----------------------------------------------
out_df = pd.DataFrame()
column_mappings = {
    'Time2': 'Time',  # Assuming 'Time2' is the correctly formatted time
    'TWC Wire Frequency (Hz)': 'TWC Wire Frequency (Hz)',
    'SLWC Wire Frequency (Hz)': 'SLWC Wire Frequency (Hz)',
    'TWC Measurement Cycles': 'TWC Measurement Cycles',
    'SLWC Measurement Cycles': 'SLWC Measurement Cycles',
    'Battery Voltage (mV)': 'Battery Voltage (mV)',
    'Altitude (km)': 'Altitude (km)',
    'Pressure (mb)': 'Pressure (mb)',
    'Temperature (C)': 'Temperature (C)',
    'RH (%)': 'RH (%)',
    'Ascent Rate (m/s)': 'Ascent Rate (m/s)',
    'GPS_wind_direction_deg': 'Wind Direction (deg)',
    'GPS_wind_speed_m_s': 'Wind Speed (m/s)'
}
# If any required column is missing in UWC2, add it as NaN.
for col in column_mappings.keys():
    if col not in UWC2.columns:
        UWC2[col] = np.nan

# Now check for missing columns (should be none)
missing_columns = [col for col in column_mappings.keys() if col not in UWC2.columns]
if missing_columns:
    raise KeyError(f"Missing required columns: {missing_columns}")

# Format the sonde time from the 'pro_time' column into 'Time2' already exists.
# Convert Time2 (string) into a uniform time format.
out_df['Time'] = UWC2['Time2']

for input_col, output_col in column_mappings.items():
    if input_col == 'Time2':
        continue
    if input_col in ['TWC Wire Frequency (Hz)', 'SLWC Wire Frequency (Hz)']:
        out_df[output_col] = pd.to_numeric(UWC2[input_col], errors='coerce')
    elif input_col in ['TWC Measurement Cycles', 'SLWC Measurement Cycles', 'Battery Voltage (mV)']:
        out_df[output_col] = pd.to_numeric(UWC2[input_col], errors='coerce').astype('Int64')
    elif input_col in ['Altitude (km)', 'Pressure (mb)', 'Temperature (C)', 'RH (%)', 'Ascent Rate (m/s)']:
        out_df[output_col] = pd.to_numeric(UWC2[input_col], errors='coerce')
    else:
        out_df[output_col] = UWC2[input_col]
out_df['Ascent Rate (m/s)'] = out_df['Ascent Rate (m/s)'].apply(lambda x: max(x, 0.0) if pd.notnull(x) else x)
desired_order = [
    'Time',
    'TWC Wire Frequency (Hz)',
    'SLWC Wire Frequency (Hz)',
    'TWC Measurement Cycles',
    'SLWC Measurement Cycles',
    'Battery Voltage (mV)',
    'Altitude (km)',
    'Pressure (mb)',
    'Temperature (C)',
    'RH (%)',
    'Ascent Rate (m/s)',
    'Wind Direction (deg)',
    'Wind Speed (m/s)'
]
out_df = out_df[desired_order]
out_df['TWC Wire Frequency (Hz)'] = (out_df['TWC Wire Frequency (Hz)'] * 1000).round().astype('Int64')
out_df['SLWC Wire Frequency (Hz)'] = (out_df['SLWC Wire Frequency (Hz)'] * 1000).round().astype('Int64')

# -----------------------------------------------
# 9. Compute Brunt–Vaisala frequency without using np.gradient()
# -----------------------------------------------
pressure = out_df['Pressure (mb)'].values * units.hPa
temperature = (out_df['Temperature (C)'].values + 273.15) * units.kelvin
theta = mpcalc.potential_temperature(pressure, temperature)
z = out_df['Altitude (km)'].values * 1000.0  # convert km to m
# Use the custom central difference function to compute dtheta/dz
dtheta_dz = central_diff(z, theta.m)
g = 9.81  # m/s²
N_squared = (g / theta.m) * dtheta_dz
N = np.where(N_squared > 0, np.sqrt(N_squared), np.nan)
out_df["Brunt-Vaisala Frequency (s^-1)"] = N

# -----------------------------------------------
# 10. Save data frame as *.csv
# -----------------------------------------------
out_df.to_csv(csv_path_out, index=False)

# -----------------------------------------------
# 11. Make skew-T plot
# -----------------------------------------------
#Trim dataframe to pressure greater than 200 hPa for plotting
pressure = out_df['Pressure (mb)'].values
pos = np.argmax(np.array(pressure)<200)
if pos==0:
    df_plot = out_df
else:
    df_plot = out_df.iloc[0:pos]     
#Create figure
fig = plt.figure(figsize=(14, 9))
pressure = df_plot['Pressure (mb)'].values * units.mbar
temperature = df_plot['Temperature (C)'].values * units.degC
dewpoint = mpcalc.dewpoint_from_relative_humidity(temperature, df_plot['RH (%)'].values / 100)
#Create Skew=t plot
skew = SkewT(fig, rotation=45, rect=(0.1, 0.1, 0.55, 0.85))
skew.plot(pressure, temperature, 'r', linewidth=2, label='Temperature')
skew.plot(pressure, dewpoint, 'g', linewidth=2, label='Dew Point')
#Add wind barbs and other skew-T lines
if ('Wind Speed (m/s)' in df_plot.columns) and ('Wind Direction (deg)' in df_plot.columns):
    u, v = mpcalc.wind_components(df_plot['Wind Speed (m/s)'].values * units('m/s'),
                                  df_plot['Wind Direction (deg)'].values * units.deg)
    skew.plot_barbs(pressure[::120], u[::120], v[::120])
skew.plot_dry_adiabats(linewidth=0.7)
skew.plot_moist_adiabats(linewidth=0.7)
skew.plot_mixing_lines(linewidth=0.7)
#Compute LCL, LFC, EL
lcl_pressure, lcl_temperature = mpcalc.lcl(pressure[0],temperature[0],dewpoint[0]) 
lfc_pressure, lfc_temperature = mpcalc.lfc(pressure,temperature,dewpoint)  
el_pressure, el_temperature = mpcalc.el(pressure,temperature,dewpoint)   
#Add LCL line to plot
skew.plot(lcl_pressure, lcl_temperature, 'ko', markerfacecolor='black')
prof = mpcalc.parcel_profile(pressure, temperature[0], dewpoint[0]).to('degC')
skew.plot(pressure, prof, 'k', linewidth=2)
#Get altitude at LCL, LFC, EL, 0°C, and -7°C
idx_lcl = (np.abs(df_plot['Pressure (mb)'].values-lcl_pressure.magnitude)).argmin()
idx_lfc = (np.abs(df_plot['Pressure (mb)'].values-lfc_pressure.magnitude)).argmin()
idx_el  = (np.abs(df_plot['Pressure (mb)'].values-el_pressure.magnitude)).argmin()
idx_0c  = (np.abs(df_plot['Temperature (C)'].values-0)).argmin()
idx_7c  = (np.abs(df_plot['Temperature (C)'].values+7)).argmin()
lcl_ev = df_plot['Altitude (km)'].values[idx_lcl] - df_plot['Altitude (km)'].values[0]
lfc_ev = df_plot['Altitude (km)'].values[idx_lfc] - df_plot['Altitude (km)'].values[0]
el_ev  = df_plot['Altitude (km)'].values[idx_el] - df_plot['Altitude (km)'].values[0]
tm0_ev = df_plot['Altitude (km)'].values[idx_0c] - df_plot['Altitude (km)'].values[0]
tm7_ev = df_plot['Altitude (km)'].values[idx_7c] - df_plot['Altitude (km)'].values[0]
#Add text for LCL, LFC, EL, and elevations
props = dict(boxstyle='round', facecolor='w', alpha=1)
skew.ax.text(0.05, 0.95, f'0°C Altitude: {round(tm0_ev,2)} km AGL',transform=skew.ax.transAxes,fontsize=12, verticalalignment='top', bbox=props)
skew.ax.text(0.05, 0.90, f'-7°C Altitude: {round(tm7_ev,2)} km AGL',transform=skew.ax.transAxes,fontsize=12, verticalalignment='top', bbox=props)
if np.isnan(lcl_pressure.magnitude):
    skew.ax.text(0.05, 0.85,'No LCL detected',transform=skew.ax.transAxes,fontsize=12, verticalalignment='top', bbox=props)
else:
    skew.ax.text(0.05, 0.85, f'LCL: {round(lcl_ev,2)} km AGL',transform=skew.ax.transAxes,fontsize=12, verticalalignment='top', bbox=props)
if np.isnan(lfc_pressure.magnitude):
    skew.ax.text(0.05, 0.80,'No LFC detected',transform=skew.ax.transAxes,fontsize=12, verticalalignment='top', bbox=props)
else:
    skew.ax.text(0.05, 0.80, f'LFC: {round(lfc_ev,2)} km AGL',transform=skew.ax.transAxes,fontsize=12, verticalalignment='top', bbox=props)
if np.isnan(el_pressure.magnitude):
    skew.ax.text(0.05, 0.75,'No EL detected',transform=skew.ax.transAxes,fontsize=12, verticalalignment='top', bbox=props)
else:
    skew.ax.text(0.05, 0.75, f'EL: {round(el_ev,2)} km AGL',transform=skew.ax.transAxes,fontsize=12, verticalalignment='top', bbox=props)
#Plot details
skew.ax.set_xlabel('Temperature (°C)')
skew.ax.set_ylabel('Pressure (hPa)')
skew.ax.set_ylim(np.max(df_plot['Pressure (mb)'].values),200)
skew.ax.set_xlim(-30, 20)
skew.ax.legend(loc="lower left")
fig.suptitle(f'Time Range: {start_time} to {end_time} UTC', fontsize=14)

# -----------------------------------------------
# 12. Make plot for SLWC and UWC frequencies
# -----------------------------------------------
#Get altitude at specific pressure levels
dfa = df_plot.dropna()
pres = dfa['Pressure (mb)'].values;
alt  = dfa['Altitude (km)'].values
P = np.arange(900, 100, -100)
count = 0; alts = np.zeros(len(P))
for i in P:
    idx = np.abs(pres - i).argmin()
    alti = np.round(alt[idx],2)
    alts[count] = alti
    count = count+1;
alts_str = list(map(str, alts))
#Make plot    
ax = plt.axes((0.7, 0.1, 0.16, 0.85))
twc_freq  = dfa['TWC Wire Frequency (Hz)'].values
slwc_freq = dfa['SLWC Wire Frequency (Hz)'].values
pres = dfa['Pressure (mb)'].values;
plt.plot(twc_freq/1000,pres,'k-',label='UWC Frequency')
plt.plot(slwc_freq/1000,pres,'y-',label='SLWC Frequency')
ax.set_xlim(5,17)
ax.set_ylim(np.max(pres),200)
ax.set_yscale('log')
plt.yticks(P,alts_str)
ax.yaxis.set_inverted(True)
plt.grid(True)
ax.set_xlabel('Frequency (MHz)')
ax.set_ylabel('Altitude (km)')
ax.legend(loc="upper center")
plt.show()
plt.close(fig)
fig.savefig(plt_path_out, dpi=300, bbox_inches='tight')



