#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 30 11:24:37 2025

@author: tylerpelle

This is a copy-cat script to re-write the gm3_folder.py script that was 
originally written by Todd to try and debug it.
"""

import os
import glob
import re
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
import metpy.calc as mpcalc
import scipy
from scipy.signal import savgol_filter
from metpy.plots import SkewT
from metpy.units import units

# -----------------------------
# Get csv and raw files
# -----------------------------
base_path = r'/Users/tylerpelle/Desktop/Rainmaker_Projects/SlwcAnalysis/TestData_NEW/'
csv_file = glob.glob(os.path.join(base_path, "*_withslw.csv"))
#csv_file =  base_path + 'cliff-0645z_20241212_withslw.csv'
csv_path_out = base_path + 'NewOutput_withslw_gm3.csv'
plt_path_out = base_path + 'NewOutput_SlwcIcePlot.png'

# ---------------------------
# Helper functions
# ---------------------------
def time_to_seconds(t):
    """Convert a HH:MM:SS time string to seconds."""
    try:
        h, m, s = t.split(':')
        return int(h) * 3600 + int(m) * 60 + int(s)
    except Exception:
        return np.nan

def central_diff(x, y):
    """
    Compute the derivative dy/dx using central differences.
    x and y must be numpy arrays of the same shape.
    The first and last points use forward and backward differences respectively.
    """
    dydx = np.empty_like(y)
    dydx[0] = (y[1] - y[0]) / (x[1] - x[0])
    dydx[-1] = (y[-1] - y[-2]) / (x[-1] - x[-2])
    for i in range(1, len(y) - 1):
        dydx[i] = (y[i + 1] - y[i - 1]) / (x[i + 1] - x[i - 1])
    return dydx

def reg_slope(subdf, col_name):
    """
    Perform a linear regression of subdf[col_name] versus subdf["time_seconds"]
    and return the slope.
    """
    x = subdf["time_seconds"].values
    y = subdf[col_name].values
    N = len(x)
    sum_x = x.sum()
    sum_y = y.sum()
    sum_x2 = (x * x).sum()
    sum_xy = (x * y).sum()
    denom = N * sum_x2 - sum_x**2
    if denom == 0:
        return np.nan
    return (N * sum_xy - sum_x * sum_y) / denom

def compute_lambda_uwc(f, Ew, Iw, Ecrmc, Icrmc, cc, L0, ac, lamw, bc, lamcrmc):
    if f == 0 or pd.isna(f):
        return 0
    return (((math.pi ** 2 * (Ew * Iw + Ecrmc * Icrmc * cc)) / (128 * (f ** 2) * (L0 ** 4))) - ac * lamw - bc * lamcrmc) / bc

def compute_lambda_slwc(f, Ew, Iw, ENi, INi, cc, L0, ac, lamw, bc, lamNi):
    if f == 0 or pd.isna(f):
        return 0
    return (((math.pi ** 2 * (Ew * Iw + ENi * INi * cc)) / (128 * (f ** 2) * (L0 ** 4))) - ac * lamw - bc * lamNi) / bc

def collection_efficiencies(riserate, dropletsize, uwcwiredia, slwcwiredia):
    """
    Given the ascent rate (riserate) and sensor geometry,
    compute the collection efficiencies for UWC and SLWC.
    """
    if riserate < 0:
        riserate = 0.0
    Rej = dropletsize * 1e-6 * riserate * 1.29 / 0.000017
    Kj_uwc = 1000 * riserate * (dropletsize * 1e-6)**2 / (9 * 0.000017 * (uwcwiredia / 1e6))
    K0j_uwc = 0.125 + ((Kj_uwc - 0.125) / (1 + 0.0967 * (Rej ** 0.6367)))
    uwccolleff = K0j_uwc / ((math.pi / 2) + K0j_uwc)
    
    Kj_slwc = 1000 * riserate * (dropletsize * 1e-6)**2 / (9 * 0.000017 * (slwcwiredia / 1e6))
    K0j_slwc = 0.125 + ((Kj_slwc - 0.125) / (1 + 0.0967 * (Rej ** 0.6367)))
    slwccolleff = K0j_slwc / ((math.pi / 2) + K0j_slwc)
    
    return uwccolleff, slwccolleff

# ---------------------------
# Read in CSV and compute required variables 
# ---------------------------
df = pd.read_csv(csv_file[0])

# Hard-coded parameters (adjust as needed)
dropletsize = 25      # expected median volume diameter in micrometers
uwcwiredia = 710      # UWC wire diameter in micrometers
slwcwiredia = 610     # SLWC wire diameter in micrometers
basewiredia = 330     # base wire diameter in micrometers

# Constants for lambda calculations
L0 = 86 / 1000        # meters
Ew = 2.05e11          # N/m^2
Ecrmc = 6.10e10
ENi = 1.15e11
ac, bc, cc = 0.2268, 0.2189, 0.1935

# Convert diameters from micrometers to meters
dw = basewiredia / 1e6
dcrmc = uwcwiredia / 1e6
dNi = slwcwiredia / 1e6

# Compute area moments of inertia
Iw = math.pi / 64 * (dw ** 4)
Icrmc = math.pi / 64 * ((dcrmc ** 4) - (dw ** 4))
INi = math.pi / 64 * ((dNi ** 4) - (dw ** 4))

# Pre-calculated linear densities
lamw_val = 0.0006
lamcrmc = 0.0015
lamNi = 0.0002

# Create a new column for time (in seconds)
df["time_seconds"] = df["Time"].apply(time_to_seconds)

# Convert frequency columns to numeric and convert to Hz
df["fuwc"] = pd.to_numeric(df["TWC Wire Frequency (Hz)"], errors='coerce') / 1000.0
df["fslwc"] = pd.to_numeric(df["SLWC Wire Frequency (Hz)"], errors='coerce') / 1000.0

# Compute lambda and collection coefficients for each row
tmpu = []; tmps = []; tmpuC = []; tmpsC = [];
for idx, row in df.iterrows():
    fuwc = row["fuwc"]
    if fuwc == 0 or pd.isna(fuwc):
        tmpu.append(0)
    else:
        tmpu.append((((math.pi ** 2 * (Ew * Iw + Ecrmc * Icrmc * cc)) / (128 * (fuwc ** 2) * (L0 ** 4))) - ac * lamw_val - bc * lamcrmc) / bc)
    fslwc = row["fslwc"]
    if fslwc == 0 or pd.isna(fslwc):
        tmps.append(0)
    else:
        tmps.append((((math.pi ** 2 * (Ew * Iw + ENi * INi * cc)) / (128 * (fslwc ** 2) * (L0 ** 4))) - ac * lamw_val - bc * lamNi) / bc)
    riserate = row["Ascent Rate (m/s)"]
    if riserate<0:
        riserate=0
    Rej = dropletsize * 1e-6 * riserate * 1.29 / 0.000017
    Kj_uwc = 1000 * riserate * (dropletsize * 1e-6)**2 / (9 * 0.000017 * (uwcwiredia / 1e6))
    K0j_uwc = 0.125 + ((Kj_uwc - 0.125) / (1 + 0.0967 * (Rej ** 0.6367)))
    tmpuC.append(K0j_uwc / ((math.pi / 2) + K0j_uwc))
    Kj_slwc = 1000 * riserate * (dropletsize * 1e-6)**2 / (9 * 0.000017 * (slwcwiredia / 1e6))
    K0j_slwc = 0.125 + ((Kj_slwc - 0.125) / (1 + 0.0967 * (Rej ** 0.6367)))
    tmpsC.append(K0j_slwc / ((math.pi / 2) + K0j_slwc))
    
df["lambda_uwc"] = tmpu
df["lambda_slwc"] = tmps
df["uwccolleff"] = tmpuC
df["slwccolleff"] = tmpsC

#Get what we need
n = 90
df_tmp = df.dropna()
slwc_freq = df_tmp["fslwc"].rolling(window=n, center=True).mean()
uwc_freq = df_tmp["fuwc"].rolling(window=n, center=True).mean()
ce_uwc = df_tmp["uwccolleff"].rolling(window=n, center=True).mean()
ce_slwc = df_tmp["slwccolleff"].rolling(window=n, center=True).mean()
riserate = df_tmp["Ascent Rate (m/s)"].rolling(window=n, center=True).mean()
wndspeed = df_tmp["Wind Speed (m/s)"].rolling(window=n, center=True).mean()
alt = df_tmp["Altitude (km)"].values

#Remove NaNs
alt = alt[~np.isnan(slwc_freq)]
uwc_ce = ce_uwc[~np.isnan(slwc_freq)]
slwc_ce = ce_slwc[~np.isnan(slwc_freq)]
riserate = riserate[~np.isnan(slwc_freq)]
wndspeed = wndspeed[~np.isnan(slwc_freq)]
uwc_freq = uwc_freq[~np.isnan(uwc_freq)]
slwc_freq = slwc_freq[~np.isnan(slwc_freq)]

# #Curve fit slwc_freq and uwc_freq
# coeff = np.polyfit(alt,slwc_freq,20)
# slwc_fit = np.polyval(coeff,alt)
# coeff = np.polyfit(alt,uwc_freq,20)
# uwc_fit = np.polyval(coeff,alt)

#Get slopes via differencing
slwc_slope = np.diff(slwc_freq)
slwc_slope = np.append(slwc_slope, slwc_slope[-1])
uwc_slope = np.diff(uwc_freq)
uwc_slope = np.append(uwc_slope, uwc_slope[-1])

#Smooth slopes out a bit
window_length = n
polyorder = 5
slwc_slope = savgol_filter(slwc_slope, window_length, polyorder)
uwc_slope = savgol_filter(uwc_slope, window_length, polyorder)

#Compute UWC
slwc_freq0 = pd.to_numeric(df.loc[800:len(df),"fslwc"], errors='coerce').mean()
uwc_freq0 = pd.to_numeric(df.loc[800:len(df),"fuwc"], errors='coerce').mean()
numer = -2*lamcrmc*1000*uwc_freq0*uwc_freq0
denom = uwc_ce*dcrmc*wndspeed*uwc_freq*uwc_freq*uwc_freq
uwc_val = np.divide(numer,denom)*uwc_slope
#Compute SLWC
numer = -2*lamNi*1000*slwc_freq0*slwc_freq0
denom = slwc_ce*dNi*wndspeed*slwc_freq*slwc_freq*slwc_freq
slwc_val = np.divide(numer,denom)*slwc_slope

#Get ICE (g/m3)

ice_val = uwc_val-slwc_val

#Make fig
fig = plt.figure(figsize=(14, 9))
plt.plot(uwc_val,alt,color='black',linewidth=2)
plt.plot(slwc_val,alt,color='blue',linewidth=2) 
plt.plot(ice_val,alt,color='gray',linewidth=2,linestyle='--') 
plt.ylim(np.min(df["Altitude (km)"].values),10);
plt.xlim(0,1)
plt.show()

#Get slopes
raise Exception("STOP HERE")

# -----------------------------
# Compute cloud classification (base, middle, top) and depth
# -----------------------------

#Compute cloud depth
cloud_depth = []
cloud_base_altitude = None

# First pass: calculate cloud depth for each row
for idx, row in df.iterrows():
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

df['Cloud Depth (km)'] = cloud_depth

# Second pass: classify each row in contiguous cloud layers
classifications = [np.nan] * len(df)
block = []  # to collect indices for a contiguous cloud segment

for i in range(len(df)):
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
df['Cloud Classification'] = classifications


# -----------------------------
# Compute UWC and SLWC concentrations based on cloud depth
# CHANGES: take abs value of slope_uwc and slope_slwc
# -----------------------------

#Compute UWC and SLWC based on where the clouds are (see above classification)
#Prealloctate rows
# df["UWC (g/m^3)"] = np.nan
# df["SLWC (g/m^3)"] = np.nan
# df["Slope SLWC"] = np.nan
# df["Slope UWC"] = np.nan
# df["Colef_slwc"] = np.nan
# df["slwc_freq_smooth"] = np.nan
# df["uwc_freq_smooth"] = np.nan

#Get cloud indicies
#idx_cldtop = np.max(df.index[df['Cloud Classification'] == 'Top'].tolist())
#idx_cldbase = np.min(df.index[df['Cloud Classification'] == 'Base'].tolist())
idx_btm = 1; idx_top = len(df)-20 #TEMPORARY
slwc_freq0 = pd.to_numeric(df.loc[324:len(df),"fslwc"], errors='coerce').mean()
uwc_freq0 = pd.to_numeric(df.loc[324:len(df),"fuwc"], errors='coerce').mean()

#Set window sizes
n = idx_top-idx_btm
window_size=10
half_window = window_size // 2
min_altitude=0


#Compute SLWC and UWC via eq. 1 (Serke et al., 2014) 
for start in range(idx_btm, idx_top, window_size):
    #Get rows (subdf) within moving window start and end indicies
    end = start + window_size
    center_index = start + half_window
    subdf = df.iloc[start:end]
    
    #Get avg quantities in subdf
    slope_uwc = np.abs(reg_slope(subdf, "lambda_uwc")) #dfdt
    slope_slwc = np.abs(reg_slope(subdf, "lambda_slwc")) #dfdt
    slope_uwc = reg_slope(subdf, "lambda_uwc") #dfdt
    slope_slwc = reg_slope(subdf, "lambda_slwc") #dfdt
    slwc_freq = pd.to_numeric(subdf["fslwc"], errors='coerce').mean() #f
    uwc_freq = pd.to_numeric(subdf["fuwc"], errors='coerce').mean() #f
    riserate = pd.to_numeric(subdf["Ascent Rate (m/s)"], errors='coerce').mean()
    wndspeed = pd.to_numeric(subdf["Wind Speed (m/s)"], errors='coerce').mean()
    ce_uwc = pd.to_numeric(subdf["uwccolleff"], errors='coerce').mean()
    ce_slwc = pd.to_numeric(subdf["slwccolleff"], errors='coerce').mean()

    if (riserate > 1) and (df.loc[center_index, "Altitude (km)"] >= min_altitude):
        #Compute UWC dNi
        numer = 2*0.0224*100*uwc_freq0*uwc_freq0*1000
        denom = ce_uwc*(uwcwiredia/1e6)*wndspeed*uwc_freq*uwc_freq*uwc_freq
        #denom = ce_uwc*(dNi/1e6)*wndspeed*uwc_freq*uwc_freq*uwc_freq
        UWC_val = np.divide(numer,denom)*slope_uwc
        #Compute SLWC
        numer = 2*0.0224*100*slwc_freq0*slwc_freq0*1000
        denom = ce_slwc*(slwcwiredia/1e6)*wndspeed*slwc_freq*slwc_freq*slwc_freq
        #denom = ce_slwc*(dNi/1e6)*wndspeed*slwc_freq*slwc_freq*slwc_freq
        SLWC_val = np.divide(numer,denom)*slope_slwc
    else:
        UWC_val = 0
        SLWC_val = 0
    df.loc[center_index, "UWC (g/m^3)"] = UWC_val
    df.loc[center_index, "SLWC (g/m^3)"] = SLWC_val
    df.loc[center_index, "Slope SLWC"] = slope_slwc
    df.loc[center_index, "Slope UWC"] = slope_uwc  
    df.loc[center_index, "Colef_slwc"] = ce_slwc
    df.loc[center_index, "Colef_uwc"] = ce_uwc
    
    #Interpolate out Nans
    df["UWC_I (g/m^3)"] = df["UWC (g/m^3)"].interpolate(method='linear')
    df["SLWC_I (g/m^3)"] = df["SLWC (g/m^3)"].interpolate(method='linear')
    df["Slope SLWC"] = df["Slope SLWC"].interpolate(method='linear')
    df["Slope UWC"] = df["Slope UWC"].interpolate(method='linear')
    df["Colef_slwc"] = df["Colef_slwc"].interpolate(method='linear')
    df["Colef_uwc"] = df["Colef_uwc"].interpolate(method='linear')
    
#Make a quick plot
fig = plt.figure(figsize=(14, 9))
plt.plot(df["UWC_I (g/m^3)"].values,df["Altitude (km)"].values); 
plt.xlim(-0.5,0.5); plt.ylim(np.min(df["Altitude (km)"].values),10);
plt.show()
df.to_csv(csv_path_out, index=False)

    
    
    

# #Compute SLWC and UWC
# for start in range(idx_btm, idx_top, window_size):
#     end = start + window_size
#     subdf = df.iloc[start:end]
#     center_index = start + half_window
#     slope_uwc = np.abs(reg_slope(subdf, "lambda_uwc"))
#     slope_slwc = np.abs(reg_slope(subdf, "lambda_slwc"))
#     #slope_uwc = reg_slope(subdf, "lambda_uwc")   #TEMP
#     #slope_slwc = reg_slope(subdf, "lambda_slwc") #TEMP
#     avg_riserate = pd.to_numeric(subdf["Ascent Rate (m/s)"], errors='coerce').mean()

#     if (avg_riserate > 1) and (df.loc[center_index, "Altitude (km)"] >= min_altitude):
#         ce_uwc = pd.to_numeric(subdf["uwccolleff"], errors='coerce').mean()
#         ce_slwc = pd.to_numeric(subdf["slwccolleff"], errors='coerce').mean()
#         #ce_uwc = df.loc[center_index, "uwccolleff"]
#         #ce_slwc = df.loc[center_index, "slwccolleff"]
#         UWC_val = slope_uwc / (ce_uwc * (uwcwiredia / 1e6) * avg_riserate) * 1000 / 1000
#         SLWC_val = slope_slwc / (ce_slwc * (slwcwiredia / 1e6) * avg_riserate) * 1000 / 1000
#     else:
#         UWC_val = 0
#         SLWC_val = 0
#     df.loc[center_index, "UWC (g/m^3)"] = UWC_val
#     df.loc[center_index, "SLWC (g/m^3)"] = SLWC_val
#     df.loc[center_index, "Slope SLWC"] = slope_slwc
#     df.loc[center_index, "Slope UWC"] = slope_uwc  
#     df.loc[center_index, "Colef_slwc"] = ce_slwc

#Set 0's on each side of cloud
if idx_btm-window_size*2<0:
    df.loc[0, "UWC (g/m^3)"] = 0
    df.loc[0, "SLWC (g/m^3)"] = 0
else:
    df.loc[idx_btm-window_size, "UWC (g/m^3)"] = 0
    df.loc[idx_btm-window_size, "SLWC (g/m^3)"] = 0
if idx_top+window_size*2>len(df):
    df.loc[-1, "UWC (g/m^3)"] = 0
    df.loc[-1, "SLWC (g/m^3)"] = 0
else:
    df.loc[idx_top+window_size, "UWC (g/m^3)"] = 0
    df.loc[idx_top+window_size, "SLWC (g/m^3)"] = 0
#Interpolate between window sizes
df["UWC_I (g/m^3)"] = df["UWC (g/m^3)"].interpolate(method='linear')
df["SLWC_I (g/m^3)"] = df["SLWC (g/m^3)"].interpolate(method='linear')
df["Slope SLWC"] = df["Slope SLWC"].interpolate(method='linear')
df["Slope UWC"] = df["Slope UWC"].interpolate(method='linear')
df["Colef_slwc"] = df["Colef_slwc"].interpolate(method='linear')

df["UWC_I (g/m^3)"] = df["UWC_I (g/m^3)"].replace(0, np.nan)
df["SLWC_I (g/m^3)"] = df["SLWC_I (g/m^3)"].replace(0, np.nan)
#Smooth a bit
#df = df[df["UWC_I (g/m^3)"] > 0]
#df = df[df["SLWC_I (g/m^3)"] > 0]
#idx = np.arange(len(df))
#df.loc[idx % 3 == 0, "UWC_I (g/m^3)"] = np.nan
#df.loc[idx % 3 == 0, "SLWC_I (g/m^3)"] = np.nan
df["UWC_I (g/m^3)"] = df["UWC_I (g/m^3)"].rolling(window=10, center=True).mean()
df["SLWC_I (g/m^3)"] = df["SLWC_I (g/m^3)"].rolling(window=10, center=True).mean()

df = df[df["Ascent Rate (m/s)"] >0]
#df.to_csv(csv_path_out, index=False)

# #Try new calculation
# #idx_btm = np.min(df.index[df['Cloud Classification'] == 'Base'].tolist())
# f0 = pd.to_numeric(df.loc[1:400,"SLWC Wire Frequency (Hz)"], errors='coerce').mean()
# f = df["SLWC Wire Frequency (Hz)"].values
# e = df["Colef_slwc"].values
# D = dNi/1e6
# w = df["Wind Speed (m/s)"].values
# dfdt = df["Slope SLWC"].values
# slwc = np.divide((2*0.0224*f0*f0),(e*D*w*f*f*f))*dfdt
# alt = df["Altitude (km)"].values
# alt0 = alt[~np.isnan(slwc)]
# slwc = slwc[~np.isnan(slwc)]


# #Make a quick plot
# fig = plt.figure(figsize=(14, 9))
# plt.plot(slwc,alt0); 
# plt.xlim(0,); plt.ylim(np.min(alt0),10);
# plt.show()











# #Get ICE
# df["ICE (g/m^3)"] =  abs(df["UWC_I (g/m^3)"] - df["SLWC_I (g/m^3)"])

# #Make figure
# fig, axs = plt.subplots(2, 2, figsize=(14, 10))
# slwc = df["SLWC_I (g/m^3)"]
# ice = df["ICE (g/m^3)"]
# alt = df["Altitude (km)"]
# maxalt = np.max(alt)
# # Bottom Left: SLWC (g/m^3) vs. Altitude
# ax3 = axs[0, 0]
# ax3.plot(slwc,alt,
#          color='purple')
# ax3.set_xlabel("SLWC (g/m^3)")
# ax3.set_ylabel("Altitude (km)")
# ax3.set_title("SLW vs. Altitude")
# ax3.set_ylim(0, maxalt)
# ax3.set_xlim(0, 1)

# # Bottom Right: ICE (g/m^3) vs. Altitude
# ax4 = axs[0, 1]
# ax4.plot(ice, alt,
#          color='black')
# ax4.set_xlabel("ICE (g/m^3)")
# ax4.set_ylabel("Altitude (km)")
# ax4.set_title("ICE vs. Altitude")
# ax4.set_xlim(0, 1)
# ax4.set_ylim(0, maxalt)
# plt.show()

