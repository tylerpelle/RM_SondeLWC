# RM_SondeLWC
Python code used to process Anasphere SLWC sonde data together with iMet Radiosonde data. The package contains two example launches to test the processing on and should be ready to use right out of the box. The code will loop through all subdirectories in the "Launches" directory to process the data and produce two figures: (1) a Skew-T Log-P plot with vertical profiles of SLWC and ICE concentrations (g/m^3); (2) A plot of the raw and filtered Anasphere frequency data (SLWC and ICE) so that we know whether or not the filtering we applied (to cut through the noise) is enough. Note that while I chose filtering parameters based on what gave relatively decent results based on all of the sondes, the filtering will likey need to be optimized on a case-by-case basis to get the best signal-to-noise ratio. The tuneable parameters that control the level of applied filtering are located at the top of the file (L72-74) and are the following: (n) The window over which the Savitzky-Golay filter is applied, higher values apply greater smoothing; (polyorder_freq) The order of the polynomial that is used to fit the Anasphere frequency data, lower values apply greater smoothing; (polyorder_slope) The order of the polynomial that is used to fit the linear density slope data, lower values apply greater smoothing.  
```
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
```
This code should be set to run right out of the box and is certainly a working version, so I expect this will be updated often as fieldwork progresses. 
