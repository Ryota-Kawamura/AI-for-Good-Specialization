# Datasheet: *SDWPF sensor data* Lab 3

Author: DeepLearning.AI (DLAI)

Files:
	wind_data.csv
	turb_location.csv

## Motivation

The dataset consists of two files.

wind_data.csv is a curated collection of mesurements from wind turbines in a wind farm in China. The dataset was used for a Spatial Dynamic Wind Power Forecasting Challenge (SDWPF) at the Baidu KDD Cup 2022 where teams competed for $35,000 in prize money. The data comes from Longyuan Power Group Corp. Ltd., which is the largest wind power producer in China and Asia.

The information about the dataset has been published on ArXiv (https://arxiv.org/abs/2208.04360) and the data is available for download online (https://aistudio.baidu.com/aistudio/competition/detail/152/0/datasets)

The curated data only includes the top 10 turbines (out of 134) in terms of power output, and has a new engineered column to identify abnormal values. The data used in this notebook contains information from the wind turbines for a range of 245 days.

weather_forecast.pkl is a pickled python dictionary, which includes wind measurements and wind forecasts for three locations around the globe. The data was taken from Open Weather's 5 day weather forecast API (https://openweathermap.org/forecast5).  

## Composition

wtbdata_245days.csv

This dataset contains measurements from sensors of 134 wind turbines. It contains the following columns (separated by commas): Datetime, TurbID, Wspd, Etmp, Itmp, Prtv, Patv, Include, WdirCos, WdirSin, NdirCos, NdirSin, PabCos, PabSin.

The Datetime, TurbID, Patv and Include columns do not have any missing values. Etmp and Itmp have one missing value. The rest of the columns have 3435 missing values each. The total number of rows is 352800 .


weather_forecast.pkl

This dataframe contains wind speed measurements and forecasts for three different global locations: Geelong, Australia, Porto Alegre, Brazil and Pittsburg, USA. For each location there is a dictionary, where the keys are times in the future (how long in advance the data was forecasted) and values are tuples of two lists: one with timestamps and one with predictions. For the key 0, the predictions are replaced by real measurements.

