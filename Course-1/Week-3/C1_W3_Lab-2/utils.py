import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
from sklearn.neighbors import KNeighborsRegressor
import json
from shapely.geometry import shape, Point, Polygon
import folium
from colour import Color
from typing import List, Tuple, Dict, Any, Optional
from datetime import datetime


FACTOR = 1.032


def parse_dms(coor: str) -> float:
    ''' Transforms strings of degrees, minutes and seconds to a decimal value
    
    Args:
        coor (str): String containing degrees in DMS format
        
    Returns:
        dec_coord (float): coordinates as a decimal value
    '''
    parts = re.split('[^\d\w]+', coor)
    degrees = float(parts[0])
    minutes = float(parts[1])
    seconds = float(parts[2]+'.'+parts[3])
    direction = parts[4]
    
    dec_coord = degrees + minutes / 60 + seconds / (3600)
    
    if direction == 'S' or direction == 'W':
        dec_coord *= -1
    
    return dec_coord


def predict_on_bogota(
    model: KNeighborsRegressor,
    n_points: int=64
) -> Tuple[np.ndarray, float, float]:
    ''' Creates a grid of predicted pollutant values based on the neighboring stations
    
    Args:
        model (KNeighborsRegressor): Model to use
        n_points (int): number of points in the grid
        
    Returns:
        predictions_xy (np.ndarray): array containing tuples of coordinates and predicted value
        dlat (float): latitude size of grid
        dlon (float): longitudinal size of grid
    '''
    with open('data/bogota.json') as f:
        js = json.load(f)

    # Check each polygon to see if it contains the point
    polygon = Polygon(shape(js['features'][0]['geometry']))
    (lon_min, lat_min, lon_max, lat_max) = polygon.bounds

    dlat = (lat_max - lat_min) / (n_points - 1)
    dlon = (lon_max - lon_min) / (n_points - 1)
    lat_values = np.linspace(lat_min - dlat, lat_max + dlat, n_points)
    lon_values = np.linspace(lon_min - dlon, lon_max + dlon, n_points)
    xv, yv = np.meshgrid(lat_values, lon_values, indexing='xy')

    predictions_xy = []

    for i in range(n_points):
        row = [0] * n_points
        for j in range(n_points):
            if polygon.contains(Point(lon_values[j], lat_values[i])):
                point = [lat_values[i], lon_values[j]]
                # Remove the data of the same station
                pred = model.predict([point])
                predictions_xy.append([lat_values[i], lon_values[j], pred[0][0]])

    predictions_xy = np.array(predictions_xy)
    
    return predictions_xy, dlat, dlon


def create_heat_map(
    predictions_xy,
    df_day: datetime,
    dlat: float,
    dlon: float,
    target_pollutant: str='PM2.5',
    popup_plots: bool=False
) -> folium.Map:
    ''' Creates a heat map of predicted pollutant values based on the neighboring stations
    
    Args:
        predictions_xy (np.ndarray): array containing tuples of coordinates and predicted value
        df_day (datetime): the day for which to show the heatmap
        dlat (float): latitude size of grid
        dlon (float): longitudinal size of grid
        target_pollutant (str): pollutant for which to show the heatmap
        popup_plots (bool): Flag whether to show plots on popup or not

    Returns:
        map_hooray (folium.Map): Heatmap on the map.
    '''
    # Create the map
    lat_center = np.average(predictions_xy[:,0])
    lon_center = np.average(predictions_xy[:,1])

    map_hooray = folium.Map(location=[lat_center, lon_center], zoom_start = 11) 

    # List comprehension to make out list of lists
    predictions = predictions_xy
    heat_data = predictions
    ymin = np.min(predictions[:,2])
    ymax = np.max(predictions[:,2])

    max_value_color = 50
    
    # Create rectangle features for the map to show the interpolated pollution between the stations
    for row in heat_data:
        color = color_producer(target_pollutant, row[2])
        folium.Rectangle(
            bounds=[
                (row[0] - dlat * FACTOR / 2, row[1] - dlon * FACTOR / 2),
                (row[0] + dlat * FACTOR / 2, row[1] + dlon * FACTOR / 2)
            ],
            color=color,
            stroke=False,
            fill=True,
            fill_color=color,
            fill_opacity=0.5,
            popup=f'{"{:.2f}".format(row[2])}'
        ).add_to(map_hooray)
    
    # Create circle features for the map to show stations
    fg = folium.FeatureGroup(name='Stations')
    for index, station in df_day.iterrows():
        imputed_col =  f'{target_pollutant}_imputed_flag'
        if imputed_col in station and type(station[imputed_col]) == str:
            bg_color = 'black'
            interpolated = f"\nestimated"
        else:
            bg_color = 'white'
            interpolated = ''
        if popup_plots:
            popup_text = f"<img src='img/tmp/{station['Station']}.png'>"
        else:
            popup_text = f"{station['Station']}:\n{'{:.2f}'.format(station[target_pollutant])}{interpolated}"
        fg.add_child(
            folium.CircleMarker(
                location=[station['Latitude'], station['Longitude']],
                radius = 11,
                fill_color=bg_color,
                color = '',
                fill_opacity=0.9,
            )
        )
        fg.add_child(
            folium.CircleMarker(
                location=[station['Latitude'], station['Longitude']],
                radius = 10,
                fill_color=color_producer(target_pollutant, station[target_pollutant]),
                color = '',
                fill_opacity=0.9,
                popup=popup_text
            )
        )
    map_hooray.add_child(fg)

    return map_hooray

    
def calculate_mae_for_k(
    data: pd.core.frame.DataFrame,
    k: int=1,
    target_pollutant: str='PM2.5'
) -> float:
    ''' Calculates the MAE for k nearest neighbors
    
    Args:
        data (pd.core.frame.DataFrame): dataframe with data.
        k (int): number of neighbors to use for interpolation
        target_pollutant (str): pollutant for which to show the heatmap

    Returns:
        MAE (float): The MAE value
    '''    
    # Drop all the rows with the stations where the data imputation didnt perform well
    bad_stations = ['7MA', 'CSE', 'COL', 'MOV2']
    df2 = data.drop(data[data['Station'].isin(bad_stations)].index)

    # Drop all the rows where there is imputed data, so the calculation is only done on real data
    # df2 = data[data[[c for c in data.columns if 'flag' in c]].isnull().all(axis=1)]
    
    # Take a sample of the data (so that the notebook runs faster)
    df2 = df2.sample(frac=0.2, random_state=8765)
    df2.insert(0, 'time_discriminator', (df2['DateTime'].dt.dayofyear * 10000 + df2['DateTime'].dt.hour * 100).values, True)
    
    predictions = []
    stations = data['Station'].unique()
    for station in stations:
        df_day_station = df2.loc[df2['Station'] == station]
        if len(df_day_station) > 0:
            df_day_no_station = df2.loc[df2['Station'] != station]
            if len(df_day_no_station) >= k:
                neigh = KNeighborsRegressor(n_neighbors=k, weights = 'distance', metric='sqeuclidean')
                knn_model = neigh.fit(
                    df_day_no_station[['Latitude', 'Longitude', 'time_discriminator']],
                    df_day_no_station[[target_pollutant]]
                )
                prediction = knn_model.predict(df_day_station[['Latitude', 'Longitude', 'time_discriminator']])
                if len(predictions)==0:
                    predictions = np.array([df_day_station[target_pollutant].values, prediction[:,0]]).T
                else:
                    predictions = np.concatenate(
                        (predictions, np.array([df_day_station[target_pollutant].values, prediction[:,0]]).T),
                        axis=0
                    )

    predictions = np.array(predictions)
    MAE = mean_absolute_error(predictions[:,0],predictions[:,1])
    
    return MAE


def create_heat_map_with_date_range(
    df: pd.core.frame.DataFrame,
    start_date: datetime,
    end_date: datetime,
    k_neighbors: int=1,
    target_pollutant: str='PM2.5',
    distance_metric: str='sqeuclidean'
) -> folium.Map:
    ''' Creates a heat map of predicted pollutant values based on the neighboring stations
    
    Args:
        df (pd.core.frame.DataFrame): dataframe with data.
        start_date (datetime): the starting day for which to show the heatmap
        end_date (datetime): the end day for which to show the heatmap
        k_neighbors (int): number of neighbors to use for interpolation
        target_pollutant (str): pollutant for which to show the heatmap
        distance_metric (str): The metric to use to calculate the distance between the stations.

    Returns:
        map_hooray (folium.Map): Heatmap on the map.
    '''
    df_days = df[df['DateTime'] >= start_date]
    df_days = df_days[df_days['DateTime'] <= end_date]

    for key in df_days.Station.unique():
        dates = df_days[df_days['Station']==key]['DateTime']
        plt.plot(dates, df_days[df_days['Station']==key][target_pollutant], '-o')
        plt.plot(dates, [12] * len(dates),'--g', label='recommended level')
        plt.title(f'Station {key}')
        plt.xlabel('hour')
        plt.ylabel(f'{target_pollutant} concentration')
        plt.legend(loc='upper left')
        plt.xticks(rotation=30)
        plt.savefig(f'img/tmp/{key}.png')
        plt.clf()

    k = k_neighbors
    neigh = KNeighborsRegressor(n_neighbors=k, weights = 'distance', metric=distance_metric)
    # Filter a single time step
    df_day = df_days[df_days['DateTime'] == end_date]
    neigh.fit(df_day[['Latitude', 'Longitude']], df_day[[target_pollutant]])

    predictions_xy, dlat, dlon = predict_on_bogota(neigh, 64)

    map_hooray = create_heat_map(predictions_xy, df_day, dlat, dlon, target_pollutant, popup_plots=True)

    return map_hooray


def create_animation_features(
    df: pd.core.frame.DataFrame,
    start_date: datetime,
    end_date: datetime,
    k: int,
    n_points: int,
    target_pollutant='PM2.5'
) -> List[Dict[str, Any]]:
    ''' Creates features to put on the animated map
    
    Args:
        df (pd.core.frame.DataFrame): dataframe with data.
        start_date (datetime): the starting day for which to show the heatmap
        end_date (datetime): the end day for which to show the heatmap
        k (int): number of neighbors to use for interpolation
        n_points (int): number of points in the grid
        target_pollutant (str): pollutant for which to show the heatmap

    Returns:
        features (List[Dict[str, Any]]): List of features.
    '''
    # Select the date range from the full dataframe
    df_days = df[df['DateTime'] >= start_date]
    df_days = df_days[df_days['DateTime'] <= end_date]
    # Take all the unique dates (steps for the animation)
    unique_dates = df_days['DateTime'].unique()
    # Select only relevant columns
    df_days = df_days[['DateTime', 'Station', 'Latitude', 'Longitude', target_pollutant]]
    # Create a list to store all of the features (elements) of the animation
    features = []

    k_neighbors_model = KNeighborsRegressor(n_neighbors=k, weights='distance', metric='sqeuclidean')

    for timestamp in unique_dates:
        df_day = df[df['DateTime'] == timestamp]

        day_hour = str(timestamp)[0:19]
        k_neighbors_model.fit(df_day[['Latitude', 'Longitude']], df_day[[target_pollutant]])
        predictions_xy, dlat, dlon = predict_on_bogota(k_neighbors_model, n_points)

        for row in predictions_xy:
            rect = create_polygon(row, dlat, dlon, day_hour, target_pollutant)
            features.append(rect)

        for index, station in df_day.iterrows():
            imputed_col =  f'{target_pollutant}_imputed_flag'
            if imputed_col in station and type(station[imputed_col]) == str:
                bg_color = 'black'
            else:
                bg_color = 'white'
            data = [station['Latitude'], station['Longitude'], station[target_pollutant]]
            circle = create_circle(data, day_hour, 13, target_pollutant, bg_color)
            features.append(circle)
            circle = create_circle(data, day_hour, 12, target_pollutant)
            features.append(circle)
    
    return features


def color_producer(pollutant_type, pollutant_value):
    ''' This function returns colors based on the pollutant values to create a color representation of air pollution.    
    
    The color scale  for PM2.5 is taken from purpleair.com and it agrees with international guidelines
    The scale for other pollutants was created based on the limits for other pollutants to approximately
    correspond to the PM2.5 color scale. The values in the scale should not be taken for granted and
    are used just for the visualization purposes.
    
    Args:
        pollutant_type (str): Type of pollutant to get the color for
        pollutant_value (float): Value of polutant concentration
        
    Returns:
        pin_color (str): The color of the bucket
    '''
    all_colors_dict = {
        'PM2.5': {0: 'green', 12: 'yellow', 35: 'orange', 55.4: 'red', 150: 'black'},
        'PM10': {0: 'green', 20: 'yellow', 60: 'orange', 110: 'red', 250: 'black'},
        'CO': {0: 'green', 4: 'yellow', 10: 'orange', 20: 'red', 50: 'black'},
        'OZONE': {0: 'green', 60: 'yellow', 100: 'orange', 200: 'red', 300: 'black'},
        'NOX': {0: 'green', 40: 'yellow', 80: 'orange', 160: 'red', 300: 'black'},
        'NO': {0: 'green', 40: 'yellow', 80: 'orange', 160: 'red', 300: 'black'},
        'NO2': {0: 'green', 20: 'yellow', 40: 'orange', 80: 'red', 200: 'black'},
    }
    
    # Select the correct color scale, if it is not available, choose PM2.5
    colors_dict = all_colors_dict.get(pollutant_type, all_colors_dict['PM2.5'])
    thresholds = sorted(list(colors_dict.keys()))
    
    previous = 0
    for threshold in thresholds:
        if pollutant_value < threshold:
            bucket_size = threshold - previous
            bucket = (pollutant_value - previous) / bucket_size
            colors = list(Color(colors_dict[previous]).range_to(Color(colors_dict[threshold]), 11))
            pin_color = str(colors[int(np.round(bucket*10))])
            return pin_color
        previous = threshold


def create_polygon(
    p: List[float],
    dlat: float,
    dlon: float,
    time: datetime,
    pollutant: str
) -> Dict[str, Any]:
    ''' Given the parameters it creates a dictionary with information for the polygon feature.
    
    Args:
        p (List[float]): list of coordinates 
        dlat (float): latitude size of grid
        dlon (float): longitudinal size of grid
        time (datetime): time for which the polygon is valid
        pollutant (str): pollutant which the polygon represents
        
    Returns:
        feature (Dict[str, Any]): dictionary of the feature properties.
    '''
    # Create a polygon feature for the map
    feature = {
            'type': 'Feature',
            'geometry': {
                'type': 'Polygon',
                'coordinates': [[[p[1] - dlon * FACTOR / 2, p[0] - dlat * FACTOR / 2], 
                                [p[1] - dlon * FACTOR / 2, p[0] + dlat * FACTOR / 2],  
                                [p[1] + dlon * FACTOR / 2, p[0] + dlat * FACTOR / 2],  
                                [p[1] + dlon * FACTOR / 2, p[0] - dlat * FACTOR / 2], 
                                [p[1] - dlon * FACTOR / 2, p[0] - dlat * FACTOR / 2]]],

            },
            'properties': {
                'times': [time], 
                'style': {
                    'color': color_producer(pollutant, p[2]), 
                    'stroke': False,
                    'fillOpacity': 0.4
                }
            }
    }
    return feature


def create_circle(
    p: List[float],
    day_hour: datetime,
    radius: float,
    pollutant: str,
    color: Optional[str]=None
) -> Dict[str, Any]:
    ''' Given the parameters it creates a dictionary with information for the circle feature.
    
    Args:
        p (List[float]): list of coordinates 
        day_hour (datetime): time for which the polygon is valid
        radius (float): size of the circle
        pollutant (str): pollutant which the polygon represents
        color (Optional[str]): color of the circle
        
    Returns:
        feature (Dict[str, Any]): dictionary of the feature properties.
    '''
    if color is None:
        color = color_producer(pollutant, p[2])
    
    feature = {
        'type': 'Feature',
        'geometry': {
            'type': 'Point',
            'coordinates': [p[1], p[0]],
        },
        'properties': {
            'time': day_hour,
            'icon': 'circle',
            'iconstyle': {
                'fillColor': color,
                'fillOpacity': 1,
                'stroke': 'false',
                'radius': radius,
            },
            'style': {'weight': 0},
        },
    }
    return feature
