#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 2, 2025 at 09:04:28
@author: Tyler Pelle @ Rainmaker Technology Corperation

This script reads in a Rawindsonde output CSV file as well as an Anasphere 
Total Liquid Water (TWC) Content raw output file and organizes the data 
into a Pandas data frame. For the TWC sensor, this script converts the raw data 
into frequency data for SLWC and UWC and then converts this frequency data into
SLWC and TWC concentrations following the methodology outlined in Heseby et al.
(2022; https://anasphere.com/documentation/UWC_Sonde_Equation_Derivation.pdf) 

Inputs:
    - Directory where launches are stored. In this package, I called this directory
      "Launches", which holds sub-directories containing the following files:
          >> Rawinsonde output file: Needs to end in "_raw.csv"
          >> Anasphere TWC sonde output file: Needs to end in ".raw"
Outputs:
    - Output *.csv file containing all computed variables (e.g., concentrations,
       fields used to compute concentrations, filtered TWC conde data, 
       cloud clasifications): Saved as "*_LwcProcessed.csv"
    - Skew-T avd SLWC/ICE concentrations plot: Saved as "*_SkewT_LWC.png"
    - Plot of raw versus filtered frequencies: Saved as "*_FreqSmoothing.png"
 
Notes: 
Previous iterations of this code employed a moving-mean to smooth over 
the frequency data and compute slopes of the linear density (which the computation
is very sensitive to. Here, we opt to filter noise from all input data using a
Savitzky-Golay filter. As opposed to a moving-mean, a Savitzky-Golay filter can
increase the signal-to-noise ratio of the data while maintaining peaks and troughs
in the inputs (whereas moving means decrease these amplitudes). This is critical 
for processing TWC sonde data because the computed concenrtations are very
sensitive to the time-derivative of the frequency data, so we thus the magnitude
of concentrations that are output are less sensitive to "how much" we filter.

Important: The level of filtering will need to be evaluated on a case-by-case basis
based on how well we cut through the noise and preserve the original signal.
However, I am confident this processing chain is providing us the correct locations
of SLWC and ICE based on the input data and the concentrations should be okay.
Below, n, polyorder_freq, and polyorder_slope are the tuneable filtering parameters.
We also produce a figure of the raw frequencies versus the filtered frequencies
so you can gauge whether these filtering parameters need to be changed.      

"""

# Import Python dependencies
import os
import glob
import re
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
import metpy.calc as mpcalc
import matplotlib.image as image
from scipy.signal import savgol_filter
from metpy.plots import SkewT
from metpy.units import units

# ---------------------------
# Set Paths and get data directories
# ---------------------------
base_path = os.getcwd() + '/Launches/'
subfolders = [f.path for f in os.scandir(base_path) if f.is_dir()]

# ---------------------------
# Set window length (n) and polynomial order for Savitzky-Golay filter
# Larger windows will smooth more, higher polynomials will include more noise
# i.e., high n and low poly will give most smoothing
# ---------------------------
n = 50
polyorder_freq  = 1
polyorder_slope = 3

# ---------------------------
# Helper Functions
# ---------------------------
# Central difference function 
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

# Convert time string (HH:MM:SS) to seconds
def time_to_seconds(t):
    """Convert a HH:MM:SS time string to seconds."""
    try:
        h, m, s = t.split(':')
        return int(h) * 3600 + int(m) * 60 + int(s)
    except Exception:
        return np.nan

# Compute linear density for UWC
def compute_lambda_uwc(f, Ew, Iw, Ecrmc, Icrmc, cc, L0, ac, lamw, bc, lamcrmc):
    if f == 0 or pd.isna(f):
        return 0
    return (((math.pi ** 2 * (Ew * Iw + Ecrmc * Icrmc * cc)) / (128 * (f ** 2) * (L0 ** 4))) - ac * lamw - bc * lamcrmc) / bc

# Compute linear density for SLWC
def compute_lambda_slwc(f, Ew, Iw, ENi, INi, cc, L0, ac, lamw, bc, lamNi):
    if f == 0 or pd.isna(f):
        return 0
    return (((math.pi ** 2 * (Ew * Iw + ENi * INi * cc)) / (128 * (f ** 2) * (L0 ** 4))) - ac * lamw - bc * lamNi) / bc

# Compute collection efficiencies for UWC and SLWC
def collection_efficiencies(riserate, dropletsize, uwcwiredia, slwcwiredia):
    """
    Given the ascent rate (riserate) and sensor geometry,
    compute the collection efficiencies for UWC and SLWC.
    """
    if riserate < 0:
        riserate = 0.0
    Rej         = dropletsize * 1e-6 * riserate * 1.29 / 0.000017
    Kj_uwc      = 1000 * riserate * (dropletsize * 1e-6)**2 / (9 * 0.000017 * (uwcwiredia / 1e6))
    K0j_uwc     = 0.125 + ((Kj_uwc - 0.125) / (1 + 0.0967 * (Rej ** 0.6367)))
    uwccolleff  = K0j_uwc / ((math.pi / 2) + K0j_uwc)
    
    Kj_slwc     = 1000 * riserate * (dropletsize * 1e-6)**2 / (9 * 0.000017 * (slwcwiredia / 1e6))
    K0j_slwc    = 0.125 + ((Kj_slwc - 0.125) / (1 + 0.0967 * (Rej ** 0.6367)))
    slwccolleff = K0j_slwc / ((math.pi / 2) + K0j_slwc)   
    return uwccolleff, slwccolleff

# ---------------------------
# Loop through subdirectories in base_path and perform processing
# ---------------------------
if not subfolders:
    raise Exception('No subfolders found in the base folder.') 

for current_folder in subfolders:  
    # ---------------------------
    # Set input and output paths
    # ---------------------------
    csv_file = glob.glob(os.path.join(current_folder, '*_raw.csv'))
    raw_file = glob.glob(os.path.join(current_folder, '*.raw'))
    base_name = os.path.splitext(os.path.basename(raw_file[0]))[0]
    csv_path_out = os.path.join(current_folder, base_name + '_LwcProcessed.csv')
    SkewT_path_out = os.path.join(current_folder, base_name + '_SkewT_LWC.png')
    FreqSmoothing_path_out = os.path.join(current_folder, base_name + '_FreqSmoothing.png')

    # ---------------------------
    # Read Anasphere LWC sensor data into data frame
    # ---------------------------
    # Read raw data into Pandas data frame
    SLW = pd.read_csv(raw_file[0], header=None, names=['Date', 'Time', 'Raw_Data'])
    SLW['Date']  = SLW['Date'].astype(str).str.strip()
    SLW['Time']  = SLW['Time'].astype(str).str.strip()
    combined_SLW = SLW['Date'] + ' ' + SLW['Time']
    combined_SLW = combined_SLW.apply(lambda x: re.sub(' +', ' ', x))
    
    # Use explicit format for SLW timestamps (adjust if needed)
    SLW['time'] = pd.to_datetime(combined_SLW, format='%Y/%m/%d %H:%M:%S.%f', errors='coerce')
    SLW.drop(['Date', 'Time'], axis=1, inplace=True)
    
    # -----------------------------
    # Read sonde data into data frame
    # -----------------------------
    sonde = pd.read_csv(
        csv_file[0],
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
    
    # Fix time
    sonde['Date']  = sonde['Date'].astype(str).str.strip()
    sonde['Time']  = sonde['Time'].astype(str).str.strip()
    combined_sonde = sonde['Date'] + ' ' + sonde['Time']
    combined_sonde = combined_sonde.apply(lambda x: re.sub(' +', ' ', x))
    
    # Use explicit format for sonde timestamps (adjust if needed)
    sonde['time'] = pd.to_datetime(combined_sonde, format='%Y-%m-%d %H:%M:%S', errors='coerce')
    sonde.drop(['Date', 'Time'], axis=1, inplace=True)
    
    # Add milliseconds for more precise time matching
    sonde['pro_time'] = sonde['time'] + pd.to_timedelta(sonde['milliseconds'], unit='ms')
    sonde = sonde.dropna(subset=['pro_time']).reset_index(drop=True)
    
    # -----------------------------
    #  Filter SLW Data by Pattern
    # -----------------------------    
    #Set pattern to search for
    pattern = '0103092901'
    
    # UWCr: SLW rows where Raw_Data contains the pattern
    UWCr = SLW[SLW['Raw_Data'].str.contains(pattern, na=False)].reset_index(drop=True)
    
    # Sonde_SLW: the row immediately following a pattern row
    Sonde_SLW = SLW.loc[SLW['Raw_Data'].str.contains(pattern, na=False).shift(-1, fill_value=False)].reset_index(drop=True)
    
    # -----------------------------------------------
    #  Merge sonde with SLW (Sonde_SLW) Data (Left Join)
    # -----------------------------------------------   
    pro_data_matched = pd.merge_asof(
        sonde.sort_values('pro_time'),
        Sonde_SLW[['time']].sort_values('time'),
        left_on   = 'pro_time',
        right_on  = 'time',
        tolerance = pd.Timedelta('2s'),
        direction = 'nearest'
    ).reset_index(drop=True)
    
    # -----------------------------------------------
    # Merge in the UWCr data to get 'Raw_Data'
    # (Left join so that every sonde record is kept;
    # if no match, 'Raw_Data' will be NaN)
    # -----------------------------------------------   
    merged_data = pd.merge_asof(
        pro_data_matched.sort_values('pro_time'),
        UWCr[['time', 'Raw_Data']].sort_values('time'),
        left_on   = 'pro_time',
        right_on  = 'time',
        tolerance = pd.Timedelta('2s'),
        direction = 'nearest'
    ).reset_index(drop=True)
    
    # -----------------------------------------------
    # Extract Fields from 'Raw_Data'
    # If no valid string is present, the fields will be NaN
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
    # Create the Combined DataFrame
    # (Use the sonde data from pro_data_matched and add the SLW fields)
    # -----------------------------------------------
    UWC2 = pd.DataFrame()
    UWC2['time']  = merged_data['pro_time']  # use sonde time as reference
    UWC2['Time2'] = UWC2['time'].dt.strftime('%H:%M:%S')
    UWC2 = pd.concat([UWC2, slw_fields], axis=1)
    UWC2['Altitude (km)']   = pro_data_matched['Altitude_km']
    UWC2['Pressure (mb)']   = pro_data_matched['Pressure_mb']
    UWC2['Temperature (C)'] = pro_data_matched['Temperature_C']
    UWC2['RH (%)']    = pro_data_matched['RH_percent']
    UWC2['Latitude']  = pro_data_matched['GPS_latitude']
    UWC2['Longitude'] = pro_data_matched['GPS_longitude']
    UWC2['Ascent Rate (m/s)']      = pro_data_matched['iMet_ascent_rate_m_s']
    UWC2['GPS_wind_speed_m_s']     = pro_data_matched['GPS_wind_speed_m_s']
    UWC2['GPS_wind_direction_deg'] = pro_data_matched['GPS_wind_direction_deg']
    UWC2 = UWC2[UWC2['GPS_wind_speed_m_s'] < 500]
    start_time = sonde.loc[sonde['iMet_ascent_rate_m_s'] > 1, 'pro_time'].min().floor('s')
    end_time = sonde['pro_time'].max().floor('s')
    
    # -----------------------------------------------
    # Create output dataframe to hold combined SLWC and sonde data
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
        raise KeyError(f'Missing required columns: {missing_columns}')
    
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
    out_df['TWC Wire Frequency (Hz)']  = (out_df['TWC Wire Frequency (Hz)'] * 1000).round().astype('Int64')
    out_df['SLWC Wire Frequency (Hz)'] = (out_df['SLWC Wire Frequency (Hz)'] * 1000).round().astype('Int64')
    
    # -----------------------------------------------
    # Compute Brunt–Vaisala frequency without using np.gradient()
    # -----------------------------------------------
    
    # Get required fields
    pressure = out_df['Pressure (mb)'].values * units.hPa
    temperature = (out_df['Temperature (C)'].values + 273.15) * units.kelvin
    theta = mpcalc.potential_temperature(pressure, temperature)
    z = out_df['Altitude (km)'].values * 1000.0  # convert km to m
    
    # Use the custom central difference function to compute dtheta/dz
    dtheta_dz = central_diff(z, theta.m)
    g = 9.81
    N_squared = (g / theta.m) * dtheta_dz
    N_squared[N_squared<0]=0    #No negatives in the square-root
    N = np.where(N_squared > 0, np.sqrt(N_squared), np.nan)
    out_df['Brunt-Vaisala Frequency (s^-1)'] = N
    
    ###############################################################################
    # Process SLWC and ICE data from Anasphere sonde
    ###############################################################################
    
    # -----------------------------
    # Hard-coded parameters (adjust as needed)
    # -----------------------------
    dropletsize = 25      # expected median volume diameter in micrometers
    uwcwiredia  = 710     # UWC wire diameter w/ gel in micrometers
    slwcwiredia = 610     # SLWC wire diameter w/ gel in micrometers
    basewiredia = 330     # Wire diameter w/out gel
    
    # Constants for lambda calculations
    L0    = 86/1000       # meters
    Ew    = 2.05e11       # N/m^2
    Ecrmc = 6.10e10
    ENi   = 1.15e11
    ac, bc, cc = 0.2268, 0.2189, 0.1935
    
    # Convert diameters from micrometers to meters
    dw    = basewiredia / 1e6
    dcrmc = uwcwiredia / 1e6
    dNi   = slwcwiredia / 1e6
    
    # Compute area moments of inertia
    Iw    = math.pi / 64 * (dw ** 4)
    Icrmc = math.pi / 64 * ((dcrmc ** 4) - (dw ** 4))
    INi   = math.pi / 64 * ((dNi ** 4) - (dw ** 4))
    
    # Pre-calculated linear densities
    lamw_val = 0.0006
    lamcrmc  = 0.0015
    lamNi    = 0.0002
    
    # -----------------------------
    # Compute needed fields
    # -----------------------------
    
    # Create a new column for time (in seconds)
    out_df['time_seconds'] = out_df['Time'].apply(time_to_seconds)
    
    # Convert frequency columns to numeric and convert to Hz
    out_df['fuwc']  = pd.to_numeric(out_df['TWC Wire Frequency (Hz)'], errors='coerce') / 1000.0
    out_df['fslwc'] = pd.to_numeric(out_df['SLWC Wire Frequency (Hz)'], errors='coerce') / 1000.0
    
    # Compute lambda for each row
    out_df['lambda_uwc']  = out_df['fuwc'].apply(lambda f: compute_lambda_uwc(f, Ew, Iw, Ecrmc, Icrmc, cc, L0, ac, lamw_val, bc, lamcrmc))
    out_df['lambda_slwc'] = out_df['fslwc'].apply(lambda f: compute_lambda_slwc(f, Ew, Iw, ENi, INi, cc, L0, ac, lamw_val, bc, lamNi))
    
    # Compute collection efficiencies for each row
    out_df['uwccolleff']  = pd.to_numeric(out_df['Ascent Rate (m/s)'], errors='coerce')\
                          .apply(lambda r: collection_efficiencies(r if r >= 0 else 0, dropletsize, uwcwiredia, slwcwiredia)[0])
    out_df['slwccolleff'] = pd.to_numeric(out_df['Ascent Rate (m/s)'], errors='coerce')\
                          .apply(lambda r: collection_efficiencies(r if r >= 0 else 0, dropletsize, uwcwiredia, slwcwiredia)[1])
    
    # -----------------------------
    # Compute SLWC, TWC, and ICE concentrations (g/m^3)
    #
    # Note that we smooth all data (freqiencies, collection efficiencies, lambdas,
    # and d(lamda)/dt) with a Savitzky-Golay filter with a window length defined
    # at the top of this script. We use this instead of a moving mean because we 
    # want to get rid of noise but preserve peaks and troughs in the signal. A 
    # moving mean will decrease peak/trough amplitudes, which will then inturn
    # decrease the magnitude of the computed concentrations. Thus, using this filter,
    # we make our processed concentrations less sensitive to the degree of smoothing.  
    # -----------------------------
    
    # Remove NaNs from dataframe
    df_tmp = out_df.dropna()
    
    # Get fields needed to compute concentrations
    slwc_freq = df_tmp['fslwc'].values
    uwc_freq  = df_tmp['fuwc'].values
    lam_uwc   = df_tmp['lambda_uwc'].values
    lam_slwc  = df_tmp['lambda_slwc'].values
    ce_uwc    = df_tmp['uwccolleff'].values
    ce_slwc   = df_tmp['slwccolleff'].values
    riserate  = df_tmp['Ascent Rate (m/s)'].values
    alt       = df_tmp['Altitude (km)'].values
    time      = df_tmp['time_seconds'].values
    
    # If there is no Anasphere sonde data, set columns to NaN
    if np.size(slwc_freq)==0:
        slwc_slope = np.nan
        uwc_slope  = np.nan
        slwc_val   = np.nan
        uwc_val    = np.nan
        ice_val    = np.nan
        uwc_freq   = np.nan
        slwc_freq  = np.nan
    
    # If there is Anasphere sonde data, compute SLWC, UWC, and ICE
    else:
        
        # Smooth data with Savitzky-Golay filter
        slwc_freq = savgol_filter(slwc_freq, n, polyorder_freq)
        uwc_freq  = savgol_filter(uwc_freq, n, polyorder_freq)
        lam_uwc   = savgol_filter(lam_uwc, n, polyorder_freq)
        lam_slwc  = savgol_filter(lam_slwc, n, polyorder_freq)
        ce_uwc    = savgol_filter(ce_uwc, n, polyorder_freq)
        ce_slwc   = savgol_filter(ce_slwc, n, polyorder_freq)
        
        # Get slopes via differencing
        slwc_slope = np.diff(lam_slwc)
        slwc_slope = np.append(slwc_slope, slwc_slope[-1])
        uwc_slope  = np.diff(lam_uwc)
        uwc_slope  = np.append(uwc_slope, uwc_slope[-1])
        
        # Further smooth the slope field (concentrations very sensitive)
        slwc_slope      = savgol_filter(slwc_slope, n, polyorder_slope)
        uwc_slope       = savgol_filter(uwc_slope, n, polyorder_slope)
                
        # Compute UWC (set negative slopes to 0, we only care when ice mass on
        # wire is increasing)
        uwc_slope[uwc_slope<0]   = 0
        slwc_slope[slwc_slope<0] = 0
        uwc_val  = uwc_slope / (ce_uwc * (uwcwiredia / 1e6) * riserate) * 1000/1000
        slwc_val = slwc_slope / (ce_slwc * (slwcwiredia / 1e6) * riserate) * 1000/1000
        
        # Get ICE (Totl minus SuperCool)
        ice_val = uwc_val-slwc_val
    
    # Create new dataframe to merge with out_df
    tmp_df = pd.DataFrame({'time_seconds': time, 'SLWC df/dt (Hz/s)': slwc_slope,
                           'UWC df/dt (Hz/s)': uwc_slope, 'SLWC (g/m3)': slwc_val, 
                           'UWC (g/m3)': uwc_val, 'ICE (g/m3)': ice_val,
                           'UWC Freq Smooth (Hz)': uwc_freq, 'SLWC Freq Smooth (Hz)': slwc_freq})
    out_df = pd.merge(out_df, tmp_df, on='time_seconds',how='left')
    
    # -----------------------------
    # Compute cloud classification (base, middle, top) and depth
    # -----------------------------

    #Compute cloud depth
    cloud_depth = []
    cloud_base_altitude = None

    # First pass: calculate cloud depth for each row
    for idx, row in out_df.iterrows():
        rh = row['RH (%)']
        altitude = row['Altitude (km)']
        if rh >= 80:
            if cloud_base_altitude is None:
                cloud_base_altitude = altitude  # mark cloud base
                cloud_depth.append(0)           # base depth is 0 km
            else:
                cloud_depth.append(altitude - cloud_base_altitude)
        else:
            cloud_depth.append(np.nan)
            cloud_base_altitude = None  # reset when out of cloud

    out_df['Cloud Depth (km)'] = cloud_depth

    # Second pass: classify each row in contiguous cloud layers
    classifications = [np.nan] * len(out_df)
    block = []  # to collect indices for a contiguous cloud segment

    for i in range(len(out_df)):
        if not pd.isna(cloud_depth[i]):
            block.append(i)
        else:
            if block:
                # Classify the contiguous cloud block
                if len(block) == 1:
                    classifications[block[0]] = 'Base'
                else:
                    classifications[block[0]] = 'Base'
                    for j in block[1:-1]:
                        classifications[j] = 'Middle'
                    classifications[block[-1]] = 'Top'
                block = []

    # If the dataframe ends with a cloud block, classify it as well.
    if block:
        if len(block) == 1:
            classifications[block[0]] = 'Base'
        else:
            classifications[block[0]] = 'Base'
            for j in block[1:-1]:
                classifications[j] = 'Middle'
            classifications[block[-1]] = 'Top'
    out_df['Cloud Classification'] = classifications
    
    # -----------------------------------------------
    # Save data frame to csv
    # -----------------------------------------------
    out_df.to_csv(csv_path_out, index=False)
    
    # -----------------------------------------------
    # Make skew-T plot and LWC plot (2x1 subplot)
    # -----------------------------------------------
    
    # Trim data to 200 hPa
    pressure = out_df['Pressure (mb)'].values
    pos = np.argmax(np.array(pressure)<200)
    if pos==0:
        df_plot = out_df
    else:
        df_plot = out_df.iloc[0:pos] 
       
    # Remove rows of dataframe w/ duplicate pressures
    df_plot = df_plot.drop_duplicates(subset=['Pressure (mb)'])
   
    # Create figure
    fig = plt.figure(figsize=(14, 9))
    
    # Get fields for plotting
    pressure    = np.sort(df_plot['Pressure (mb)'].values)
    pressure    = pressure[::-1] * units.mbar
    temperature = df_plot['Temperature (C)'].values * units.degC
    dewpoint    = mpcalc.dewpoint_from_relative_humidity(temperature, df_plot['RH (%)'].values / 100)
    
    # Create skew-T plot
    skew = SkewT(fig, rotation=45, rect=(0.1, 0.1, 0.55, 0.85))
    skew.plot(pressure, temperature, 'r', linewidth=2, label='Temperature')
    skew.plot(pressure, dewpoint, 'g', linewidth=2, label='Dew Point')
    
    # Add wind barbs and other adiabats
    if ('Wind Speed (m/s)' in df_plot.columns) and ('Wind Direction (deg)' in df_plot.columns):
        u, v = mpcalc.wind_components(df_plot['Wind Speed (m/s)'].values * units('m/s'),
                                      df_plot['Wind Direction (deg)'].values * units.deg)
        skew.plot_barbs(pressure[::120], u[::120], v[::120])
    skew.plot_dry_adiabats(linewidth=0.7)
    skew.plot_moist_adiabats(linewidth=0.7)
    skew.plot_mixing_lines(linewidth=0.7)
    
    # Compute LCL, LFC, EL
    lcl_pressure, lcl_temperature = mpcalc.lcl(pressure[0],temperature[0],dewpoint[0]) 
    lfc_pressure, lfc_temperature = mpcalc.lfc(pressure,temperature,dewpoint)
    el_pressure, el_temperature   = mpcalc.el(pressure,temperature,dewpoint)   
    
    # Add LCL line to plot
    skew.plot(lcl_pressure, lcl_temperature, 'ko', markerfacecolor='black')
    prof = mpcalc.parcel_profile(pressure, temperature[0], dewpoint[0]).to('degC')
    skew.plot(pressure, prof, 'k', linewidth=2)
    
    # Get altitude at LCL, LFC, EL, 0°C, and -7°C
    idx_lcl = (np.abs(df_plot['Pressure (mb)'].values-lcl_pressure.magnitude)).argmin()
    idx_lfc = (np.abs(df_plot['Pressure (mb)'].values-lfc_pressure.magnitude)).argmin()
    idx_el  = (np.abs(df_plot['Pressure (mb)'].values-el_pressure.magnitude)).argmin()
    idx_0c  = (np.abs(df_plot['Temperature (C)'].values-0)).argmin()
    idx_7c  = (np.abs(df_plot['Temperature (C)'].values+7)).argmin()
    lcl_ev  = df_plot['Altitude (km)'].values[idx_lcl] - df_plot['Altitude (km)'].values[0]
    lfc_ev  = df_plot['Altitude (km)'].values[idx_lfc] - df_plot['Altitude (km)'].values[0]
    el_ev   = df_plot['Altitude (km)'].values[idx_el]  - df_plot['Altitude (km)'].values[0]
    tm0_ev  = df_plot['Altitude (km)'].values[idx_0c]  - df_plot['Altitude (km)'].values[0]
    tm7_ev  = df_plot['Altitude (km)'].values[idx_7c]  - df_plot['Altitude (km)'].values[0]
    
    # Add text for LCL, LFC, EL, and elevations
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
    
    # Plot details
    skew.ax.set_xlabel('Temperature (°C)')
    skew.ax.set_ylabel('Pressure (hPa)')
    skew.ax.set_ylim(np.max(df_plot['Pressure (mb)'].values),200)
    skew.ax.set_xlim(-30, 20)
    skew.ax.legend(loc='lower left')
    plt.title(f'Time Range: {start_time} to {end_time} UTC', fontsize=14)
    
    # Add Rainmaker logo in the upper right of the skew-T plot
    logo  = image.imread(os.getcwd() + '/Rain.png')
    newax = fig.add_axes([0.51, 0.82, 0.1, 0.1])  
    newax.imshow(logo)
    newax.axis('off')
    
    # Make plot for SLWC, ICE, and relative humidity
    dfa = df_plot.dropna(subset=['SLWC (g/m3)'])
   
    # If we don't have Anasphere sonde data, only plot relative humidity  
    if dfa.empty:
        
        # Get altitudes for specific pressure levels
        pres = out_df['Pressure (mb)'].values
        alt  = out_df['Altitude (km)'].values
        rh   = out_df['RH (%)'].values
        alt  = alt-alt[0]  #Alt. above ground level
        P    = [pres[0]]
        if round(pres[0],-2)>pres[0]:
             P.extend(np.arange(round(pres[0],-2)-100, 100, -100))
        else:
             P.extend(np.arange(round(pres[0],-2), 100, -100))
        count = 0; alts = np.zeros(len(P))
        for i in P:
            idx    = np.abs(pres - i).argmin()
            alti   = np.round(alt[idx],2)
            alts[count] = alti
            count  = count+1
        alts[2:-1] = np.round([alts[2:-1]-0.26],2)
        alts_str   = list(map(str, alts))
        
        # Creat axes for figure
        ax = plt.axes((0.7, 0.1, 0.16, 0.85))
        
        # Plot
        ax.plot(rh,pres,'g-')
        
        # Plot details
        ax.spines['top'].set_color('g')
        ax.tick_params(axis='x', colors='g')
        ax.set_ylim(np.max(pres),200)
        ax.set_yscale('log')
        plt.yticks(P,alts_str)
        ax.set_xlim(0,100)
        ax.set_ylim(min(P),max(P))
        ax.yaxis.set_inverted(True)
        plt.grid(True)
        ax.set_xlabel('Relatve Humidity (%)',color='g')
        
        #Save
        fig.savefig(SkewT_path_out, dpi=300, bbox_inches='tight')
    
    # If we have Anasphere sonde data, then plot all          
    else:
        
        # Get altitudes for specific pressure levels
        pres = out_df['Pressure (mb)'].values
        alt  = out_df['Altitude (km)'].values
        alt  = alt-alt[0]  #Alt. above ground level
        P    = [pres[0]]
        if round(pres[0],-2)>pres[0]:
             P.extend(np.arange(round(pres[0],-2)-100, 100, -100))
        else:
             P.extend(np.arange(round(pres[0],-2), 100, -100))
        count = 0; alts = np.zeros(len(P))
        for i in P:
            idx    = np.abs(pres - i).argmin()
            alti   = np.round(alt[idx],2)
            alts[count] = alti
            count  = count+1
        alts[2:-1] = np.round([alts[2:-1]-0.26],2)
        alts_str   = list(map(str, alts))
       
        # Get fields for plotting
        slwc_val = out_df['SLWC (g/m3)'].interpolate(method='linear').values
        ice_val  = out_df['ICE (g/m3)'].interpolate(method='linear').values
        num      = np.shape(np.where(np.isnan(slwc_val)))[1]
        
        # Make plot (SWLC and ICE)   
        ax = plt.axes((0.7, 0.1, 0.16, 0.85))
        
        # Plot
        plt.plot(slwc_val,pres,'y-',linewidth=2,label='SLWC (g/m^3)')
        plt.plot(ice_val,pres,'k-',linewidth=2,label='ICE (g/m^3)')
        
        # Plot details
        ax.set_ylim(np.max(pres),200)
        ax.set_yscale('log')
        plt.yticks(P,alts_str)
        ax.set_xlim(0,0.2)
        ax.set_ylim(min(P),max(P))
        ax.yaxis.set_inverted(True)
        ax.set_xlabel('Concentration (g/m^3)')
        ax.legend(loc='upper center',facecolor=[0.9,0.9,0.9])
        ax.set_ylabel('Altitude (km AGL)')
        
        # Plot relative humidity (upper-x axis in green)
        rh   = out_df['RH (%)'].values
        alt  = out_df['Altitude (km)'].values
        pres = out_df['Pressure (mb)'].values
        alt  = alt-alt[0]  #Alt. above ground level
        ax2  = ax.twiny()
        ax2.plot(rh,pres,'g-')
        ax2.spines['top'].set_color('g')
        ax2.tick_params(axis='x', colors='g')
        plt.grid(True)
        ax2.set_xlabel('Relatve Humidity (%)',color='g')
        
        #Save
        fig.savefig(SkewT_path_out, dpi=300, bbox_inches='tight')
        
        # -----------------------------------------------
        # Plot Raw and Smoothed LWC frequency data
        # -----------------------------------------------
        
        # Get data
        uwc_freq_smooth = dfa['UWC Freq Smooth (Hz)'].values
        slwc_freq_smooth = dfa['SLWC Freq Smooth (Hz)'].values
        uwc_freq = dfa['fuwc'].values
        slwc_freq = dfa['fslwc'].values
        alt = dfa['Altitude (km)'].values
        alt = alt-alt[0]  #Alt. above ground level
        
        # Create fig and plot
        fig, (ax1, ax2) = plt.subplots(1,2, figsize=(8, 6))
        ax1.plot(slwc_freq,alt,'y-',linewidth=2)
        ax1.plot(uwc_freq,alt,'k-',linewidth=2)
        
        # Plot un-filtered frequencies
        ax1.set_xlabel('Frequency (Hz)')
        ax1.set_ylabel('Altitude (km AGL)') 
        ax1.set_title('Raw Frequencies')
        ax1.grid(True)
        ax1.set_xlim(6,25)
        ax1.set_ylim(min(alt),max(alt))
        
        # Plot filtered frequencies
        ax2.plot(slwc_freq_smooth,alt,'y-',linewidth=2,label='SLWC')
        ax2.plot(uwc_freq_smooth,alt,'k-',linewidth=2, label='UWC')
        ax2.set_xlabel('Frequency (Hz)') 
        ax2.set_title('Smoothed Frequencies')
        ax2.set_xlim(6,25)
        ax2.set_ylim(min(alt),max(alt))
        ax2.legend(loc="upper left",facecolor=[0.9,0.9,0.9])
        ax2.grid(True)
        
        # Show plot and save
        plt.show()
        plt.close(fig)
        fig.savefig(FreqSmoothing_path_out, dpi=300, bbox_inches='tight')

    
