# Datasheet: *RMCAB sensor data* Lab 3

Author: DeepLearning.AI (DLAI)

Files:
	full_data_with_imputed_values.csv

## Motivation

The dataset is a collection of mesurements of various pollutants at several measurement stations across Bogotá, Colombia. The dataset is based on two files (RMCAB_air_quality_sensor_data.csv, stations_loc.csv) that were downloaded to be used in the DLAI course "AI for Public Health". 

The datasets were downloaded from the public portal on the Red de Monitoreo de Calidad del Aire de Bogotá (RMCAB) website
http://201.245.192.252:81/home/map and RMCAB staff also gave explicit permission for the data to be used in this online course.

The two datasets were modified and merged to a single file in the previous lab of this course (Air Quality: Design your Solution) and missing values were imputed using various techniques.


## Composition

The dataset contains the measurements for various stations in Bogotá throughout 2021. The dataset includes the following columns: DateTime, Station, Latitude, Longitude, PM2.5, PM10, NO, NO2, NOX, CO, OZONE, PM2.5_imputed_flag, PM10_imputed_flag, NO_imputed_flag, NO2_imputed_flag, NOX_imputed_flag, CO_imputed_flag, OZONE_imputed_flag.

DateTime, Station, Latitude and Longitude columns are fully populated with original data and represent metadata (time and location) for each measurement (row). Each row represents a single measurement at the given station and time.

The pollutant columns (PM2.5, PM10, NO, NO2, NOX, CO, OZONE) are also fully populated, where some of the rows include original (raw) data, while other rows have data imputed either with interpolation or with a neural network. Each pollutant column has its corresponding flag column (columns ending with "imputed_flag") where each row independently tells whether the value for the pollutant in the given row is imputed or original and which method wasu used for imputation. In case the flag column is empty, the data in the corresponding pollutant column is original.

The dataset has 166441 rows in total.
