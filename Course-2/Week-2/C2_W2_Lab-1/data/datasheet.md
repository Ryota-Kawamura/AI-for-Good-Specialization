# Datasheet: *SDWPF sensor data* Lab 1

Author: DeepLearning.AI (DLAI)

Files:
	wtbdata_245days.csv
	turb_location.csv

## Motivation

The dataset is a collection of mesurements from 134 wind turbines in a wind farm in China. The dataset was used for a Spatial Dynamic Wind Power Forecasting Challenge (SDWPF) at the Baidu KDD Cup 2022 where teams competed for $35,000 in prize money. The data comes from Longyuan Power Group Corp. Ltd., which is the largest wind power producer in China and Asia.

The information about the dataset has been published on ArXiv (https://arxiv.org/abs/2208.04360) and the data is available for download online (https://aistudio.baidu.com/aistudio/competition/detail/152/0/datasets)

The data used in this notebook contains information from the wind turbines for a range of 245 days and consists of two separate .csv files.

## Composition

wtbdata_245days.csv

This dataset contains measurements from sensors of 134 wind turbines. It contains the following columns (separated by commas): TurbID, Day, Tmstamp, Wspd (m/s), Wdir (°), Etmp (°C), Itmp (°C), Ndir (°), Pab1 (°), Pab2 (°), Pab3 (°), Prtv (kW), Patv (kW).

The columns contain the following information:
TurbID: Wind turbine identification number.
Day: The number of the day represented as a string (first day is May 1st 2020).
Tmstamp: The hour and minute of the date of the measurement.
Wspd: The wind speed recorded by the anemometer measured in meters per second.
Wdir: The angle between the wind direction and the position of turbine nacelle measured in degrees.
Etmp: Temperature of the surounding environment measured in degrees Celsius.
Itmp: Temperature inside the turbine nacelle measured in degrees Celsius.
Ndir: Nacelle direction, i.e., the yaw angle of the nacelle measured in degrees.
Pab1: Pitch angle of blade 1 measured in degrees.
Pab2: Pitch angle of blade 2 measured in degrees.
Pab3: Pitch angle of blade 3 measured in degrees.
Prtv: Reactive power measured in kW.
Patv: Active power measured in kW

The TurbID, Day and Tmstamp columns do not have any missing values. The rest of the columns have 49518 missing values each. The total number of rows is 4727520.


turb_location.csv

This dataframe contains locations of the turbines. It has three columns: TurbID, x, y. TurbID contains integers from 1 to 134, which are the unique ID's of the turbines. x and y columns contain locations. There are 134 rows in total and no missing data.

