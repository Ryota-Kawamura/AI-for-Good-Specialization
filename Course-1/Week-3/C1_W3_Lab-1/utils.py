import numpy as np
import re
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
import math
from sklearn.model_selection import train_test_split
import warnings
import ipywidgets as widgets
from ipywidgets import interact
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer
from typing import List, Tuple, Dict
from datetime import datetime


FONT_SIZE_TICKS = 14
FONT_SIZE_TITLE = 20
FONT_SIZE_AXES = 16


# All of the functions defined below are directly called in the notebook
pollutants_list = ['PM2.5','PM10','NO','NO2','NOX','CO','OZONE']


def fix_dates(df: pd.core.frame.DataFrame, date_column: str) -> List[str]:
    '''Fixes the date format in the dataframe.

    Args:
        df (pd.core.frame.DataFrame): The dataframe.
        date_column (str): Column with dates
        
    Returns:
        fixed_dates (List[str]): list of corrected dates to be put into the dataframe
    '''
    dates = df[date_column]
    fixed_dates = []
    for row in dates:
        line = list(row)
        hour = int(''.join(line[11:13])) - 1
        fixed_dates.append("".join(line[:11] + [str(int(hour/10)) + str(int(hour % 10))] + line[13:]))
    return fixed_dates


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


def create_time_series_plot(df: pd.core.frame.DataFrame, start_date: datetime, end_date: datetime):
    '''Creates a time series plot, showing the concentration of pollutants over time.
    The pollutant and the measuring station can be selected with a dropdown menu.
    If the dataframe includes the imputed values, it will plot them in red color.
    
    Args:
        df (pd.core.frame.DataFrame): The dataframe with the data.
        start_date (str): minimum date for plotting.
        end_date (str): maximum date for plotting.
    
    '''
    def _interactive_time_series_plot(station, date_range, target):
        plt.figure(figsize=(15,6))
        
        data = df[df.Station==station]
        data = data[data.DateTime > date_range[0]]
        data = data[data.DateTime < date_range[1]]
        
        if f'{target}_imputed_flag' in data:
            # If there is imputed flag, separate the data and plot in two colors
            imputed_data = data[['DateTime', target, f'{target}_imputed_flag']]
            imputed_data.loc[imputed_data[f'{target}_imputed_flag'].isnull(), target] = None 
            original_data = data[['DateTime', target, f'{target}_imputed_flag']]
            original_data.loc[imputed_data[f'{target}_imputed_flag'].notnull(), target] = None
            plt.plot(imputed_data["DateTime"], imputed_data[target], 'r-', label='Imputed')
            plt.plot(original_data["DateTime"], original_data[target], '-', label='Real')
            plt.legend()
        else:
            # Plot the data
            plt.plot(data["DateTime"], data[target], '-', label='Real')
        
        plt.title(f'Temporal change of {target}', fontsize=FONT_SIZE_TITLE)
        plt.ylabel(f'{target} concentration', fontsize=FONT_SIZE_AXES)
        plt.xticks(rotation=20, fontsize=FONT_SIZE_TICKS)
        plt.yticks(fontsize=FONT_SIZE_TICKS)
        plt.show()
    
    # Widget for picking the city
    station_selection = widgets.Dropdown(
        options=df.Station.unique(),
        description='Station'
    )

    target_pollutant_selection = widgets.Dropdown(
        options=pollutants_list,
        description='Pollutant',
        value='PM2.5'
    )
    
    dates = pd.date_range(start_date, end_date, freq='D')

    options = [(date.strftime(' %d/%m/%Y '), date) for date in dates]
    index = (0, len(options)-1)

    selection_range_slider = widgets.SelectionRangeSlider(
        options=options,
        index=index,
        description='Dates',
        orientation='horizontal',
        layout={'width': '500px'}
    )

    # Putting it all together
    interact(
        _interactive_time_series_plot,
        station=station_selection,
        target=target_pollutant_selection,
        date_range=selection_range_slider    
    )

def plot_distribution_of_gaps(df: pd.core.frame.DataFrame, target: str):
    '''Plots the distribution of the gap sizes in the dataframe
    
    Args:
        df (pd.core.frame.DataFrame): The dataframe
        target (str): The chosen pollutant for which it plots the distribution
    '''
    def get_size_down_periods(df, target):
        '''Get the size of the downtime periods for the sensor'''
        distribution = [0] * 4000
        x = []
        i = -1
        total_missing = 0
        count = 0
        for row in df[target].values:
            if math.isnan(row):
                total_missing += 1
                if i == 0:
                    count = 1
                    i = 1
                else:
                    count += 1
            else:
                try:
                    if count > 0:
                        distribution[count] += 1 
                        x.append(count)
                except:
                    print(count)
                i = 0
                count = 0

        distribution[0] = df[target].shape[0] - total_missing

        return distribution

    distribution = get_size_down_periods(df, target=target)
    for i in range(len(distribution)):
        distribution[i] = distribution[i]*i
    only_missing_per = distribution[1:-1]
    
    plt.figure(figsize=(10,6))
    plt.plot(only_missing_per)
    plt.xlabel('Gap size (Hours)', fontsize=FONT_SIZE_AXES)
    plt.ylabel('Number of missing data points', fontsize=FONT_SIZE_AXES)
    plt.title('Distribution of gaps in the data', fontsize=FONT_SIZE_TITLE)
    plt.xticks(fontsize=FONT_SIZE_TICKS)
    plt.yticks(fontsize=FONT_SIZE_TICKS)
    plt.show()


def visualize_missing_values_estimation(df: pd.core.frame.DataFrame, day: datetime):
    '''Visualizes two ways of interpolating the data: nearest neighbor and last value
    and compares them to the real data
    
    Args:
        df (pd.core.frame.DataFrame): The dataframe
        day (datetime): The chosen day to plot
    '''
    day = day.date()

    # Filter out the data for the day for the USM station
    rows_of_day = df.apply(lambda row : row['DateTime'].date() == day, axis=1)
    sample = df[rows_of_day]
    
    def draw(sample, station, missing_index, target):
        sample = sample.copy()
        sample.insert(
            0,
            'time_discriminator', 
            (sample['DateTime'].dt.dayofyear * 100000 + sample['DateTime'].dt.hour * 100).values,
            True
        )

        real = sample[sample['Station'] == station]
        example1 = real.copy()
        real = real.reset_index()
        example1 = example1.reset_index()
        example1.loc[missing_index, target] = float('NaN')

        missing = missing_index
        missing_before_after = [missing[0]-1] + missing + [missing[-1] + 1]
        dates = set(list(example1.loc[missing_index,'DateTime'].astype(str)))

        plt.figure(figsize=(10, 5))
        plt.plot(missing_before_after, real.loc[missing_before_after][target] , 'r--o', label='actual values')

        sample_copy = sample.copy()
        sample_copy = sample_copy.reset_index()
        to_nan = sample_copy.apply(lambda row : str(row['DateTime']) in dates and row['Station'] == station, axis=1)
        sample_copy.loc[to_nan, target] = float('NaN')
        imputer = KNNImputer(n_neighbors=1)
        imputer.fit(sample_copy[['time_discriminator','Latitude', 'Longitude', target]])
        example1[f'new{target}'] = imputer.transform(example1[['time_discriminator', 'Latitude', 'Longitude', target]])[:,3]
        plt.plot(missing_before_after, example1.loc[missing_before_after][f'new{target}'], 'g--o', label='nearest neighbor')

        plt.plot(example1.index, example1[target], '-*')

        example1[f'ffill{target}'] = example1.fillna(method='ffill')[target]
        plt.plot(missing_before_after, example1.loc[missing_before_after][f'ffill{target}'], 'y--*', label='last known value')

        plt.xlabel('Hour of day', fontsize=FONT_SIZE_AXES)
        plt.ylabel(f'{target} concentration', fontsize=FONT_SIZE_AXES)
        plt.title('Estimating missing values', fontsize=FONT_SIZE_TITLE)
        plt.legend(loc="upper left", fontsize=FONT_SIZE_TICKS)
        plt.xticks(fontsize=FONT_SIZE_TICKS)
        plt.yticks(fontsize=FONT_SIZE_TICKS)
        plt.show()
    
    def selector(station, hour_start, window_size, target):
        missing_index_list = list(range(hour_start, hour_start+window_size)) 
        draw(
            sample=sample,
            station=station,
            missing_index=missing_index_list,
            target=target
        )
    
    # Widgets for selecting the parameters
    station_selection = widgets.Dropdown(
        options=df.Station.unique(),
        description='Station',
        value='USM'
    )
    target_pollutant_selection = widgets.Dropdown(
        options=pollutants_list,
        description='Pollutant',
        value='PM2.5'
    )
    hour_start_selection = widgets.Dropdown(
        options=list([2, 3, 4, 5, 6, 7, 8, 9, 10]),
        description='Hour start',
        value=3
    )
    window_size_selection = widgets.Dropdown(
        options=list([1, 2, 3, 5, 6, 9, 12]),
        description='Window size',
        value=1
    )
    
    return interact(
        selector,
        station=station_selection,
        hour_start=hour_start_selection,
        window_size=window_size_selection,
        target=target_pollutant_selection
    )


def calculate_mae_for_nearest_station(df: pd.core.frame.DataFrame, target: str) -> Dict[str, float]:
    '''Create a nearest neighbor model and run it on your test data
    
    Args:
        df (pd.core.frame.DataFrame): The dataframe
        target (str): The chosen pollutant for which it plots the distribution
    '''
    df2 = df.dropna(inplace=False)
    df2.insert(0, 'time_discriminator', (df2['DateTime'].dt.dayofyear * 100000 + df2['DateTime'].dt.hour * 100).values, True)

    train_df, test_df = train_test_split(df2, test_size=0.2, random_state=57)

    imputer = KNNImputer(n_neighbors=1)
    imputer.fit(train_df[['time_discriminator','Latitude', 'Longitude', target]])

    regression_scores = {}

    y_test = test_df[target].values

    test_df2 = test_df.copy()
    test_df2.loc[test_df.index, target] = float("NAN")

    y_pred = imputer.transform(test_df2[['time_discriminator', 'Latitude', 'Longitude', target]])[:,3]
    
    return {"MAE": mean_absolute_error(y_pred, y_test)}    


def build_keras_model(input_size: int) -> tf.keras.Model:
    '''Build a neural network with three fully connected layers (sizes: 64, 32, 1)
    
    Args:
        input_size (int): The size of the input
        
    Returns:
        model (tf.keras.Model): The neural network
    '''
    model = keras.Sequential([
        layers.Dense(64, activation='relu', input_shape=[input_size]),
        layers.Dense(32, activation='relu'),
        layers.Dense(1)
      ])

    optimizer = tf.keras.optimizers.RMSprop(0.007)

    model.compile(loss='mse',
                optimizer=optimizer,
                metrics=['mae'])
    
    return model


def train_and_test_model(
    feature_names: List[str],
    target: str,
    train_df: pd.core.frame.DataFrame,
    test_df: pd.core.frame.DataFrame,
    model: tf.keras.Model,
    number_epochs: int=100
) -> Tuple[tf.keras.Model, StandardScaler, Dict[str, float]]:
    '''
    This function will take the features (x), the target (y) and the model and will fit
    and Evaluate the model.
    
    Args:
        feature_names (List[str]): Names of feature columns
        target (str): Name of the target column
        train_df (pd.core.frame.DataFrame): Dataframe with training data
        test_df (pd.core.frame.DataFrame): Dataframe with test data
        model (tf.keras.Model): Model to be fit to the data
        number_epochs (int): Number of epochs
    
    Returns:
        model (tf.keras.Model): Fitted model
        scaler (StandardScaler): scaler
        MAE (Dict[str, float]): Dictionary containing mean absolute error.
    '''
    scaler = StandardScaler()
    
    X_train = train_df[feature_names]
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    y_train = train_df[target]
    X_test = test_df[feature_names]
    X_test = scaler.transform(X_test)
    y_test = test_df[target]

    # Build and train model
    model.fit(X_train, y_train, batch_size=64, epochs=number_epochs)
    y_pred = model.predict(X_test)
    #print(f"\nModel Score: {model.score(X_test, y_test)}")
    MAE = {"MAE": mean_absolute_error(y_pred, y_test)}
    return model, scaler, MAE


def create_plot_with_preditions(
    df: pd.core.frame.DataFrame,
    model: tf.keras.Model,
    scaler: StandardScaler,
    feature_names: List[str],
    target: str,
    start_date: datetime,
    end_date: datetime
):
    '''
    This function will take the features (x), the target (y) and the model and will fit
    and Evaluate the model.
    
    Args:
        df (pd.core.frame.DataFrame): The dataframe with the data.
        model (tf.keras.Model): Model
        scaler (StandardScaler): scaler
        feature_names (List[str]): Names of feature columns
        target (str): Name of the target column
        start_date (str): minimum date for plotting.
        end_date (str): maximum date for plotting.
    '''
    def draw_example3(sample, station, predicted2, missing_index):
        sample = sample.copy()
        sample.insert(0, 'time_discriminator', 
                      (sample['DateTime'].dt.dayofyear * 100000 + sample['DateTime'].dt.hour * 100).values, True)
        
        real_data = sample[sample['Station'] == station]
        example1 = real_data.copy()
        real_data = real_data.reset_index()
        example1 = example1.reset_index()
        example1.loc[missing_index, target] = float('NaN')

        missing = missing_index
        missing_before_after = [missing[0]-1] + missing + [missing[-1] + 1]
        dates = set(list(example1.loc[missing_index,'DateTime'].astype(str)))


        plt.plot(missing_before_after, real_data.loc[missing_before_after][target] , 'r--o', label='actual values')

        copy_of_data = sample.copy()
        copy_of_data = copy_of_data.reset_index()
        to_nan = copy_of_data.apply(lambda row : str(row['DateTime']) in dates and row['Station'] == station, axis=1)
        copy_of_data.loc[to_nan, target] = float('NaN')

        imputer = KNNImputer(n_neighbors=1)
        imputer.fit(copy_of_data[['time_discriminator','Latitude', 'Longitude', target]])
        example1[f'new_{target}'] = imputer.transform(example1[['time_discriminator', 'Latitude', 'Longitude', target]])[:,3]
        
        plt.plot(missing_before_after, example1.loc[missing_before_after][f'new_{target}'], 'g--o', label='nearest neighbor')
        plt.plot(example1.index, example1[target], '-*')

        example1[f'nn_{target}'] = example1[target].copy()
        example1.loc[missing, f'nn_{target}'] = predicted2[np.array(missing)]
        plt.plot(missing_before_after, example1.loc[missing_before_after][f'nn_{target}'], 'y--*', label='neural network')

        plt.xlabel('Index', fontsize=FONT_SIZE_AXES)
        plt.ylabel(f'{target} concentration', fontsize=FONT_SIZE_AXES)
        plt.title('2 days data and predictions', fontsize=FONT_SIZE_TITLE)
        plt.legend(loc="upper left", fontsize=FONT_SIZE_TICKS)
        plt.xticks(fontsize=FONT_SIZE_TICKS)
        plt.yticks(fontsize=FONT_SIZE_TICKS)
        
        
    def plot_predictions(station, size, start_index):
        try:
            data = df[df.DateTime > start_date]
            data = data[data.DateTime < end_date]

            X_test = data[df.Station==station]
            X_test = X_test[feature_names]
            X_test = scaler.transform(X_test)
            y_test = data[target]

            y_predicted = model.predict(X_test)

            plt.figure(figsize=(10, 5))
            draw_example3(data, station, y_predicted, list(range(start_index, start_index + size)))
            plt.show()
        except Exception as e:
            print('The selected range cannot be plotted due to missing values. Please select other values.\n')
            print(e)
    
    # Widget for picking the station
    station_selection = widgets.Dropdown(
        options=df.Station.unique(),
        description='Station'
    )
    # Widget for picking the window size
    windows_size_selection = widgets.Dropdown(
        options=list([1, 2, 3, 5, 6, 12, 24]),
        description='Window'
    )
    # Widget for selecting index of data
    index_selector = widgets.IntSlider(value=1,
                                       min=1,
                                       max=48,
                                       step=1, 
                                       description='Index')

    interact(plot_predictions, station=station_selection, size=windows_size_selection, start_index = index_selector)
    

def impute_nontarget_missing_values_interpolate(
    df_with_missing: pd.core.frame.DataFrame,
    feature_names: List[str],
    target: str,
) -> pd.core.frame.DataFrame:
    '''
    Imputes data to non-target variables using interpolation.
    This data can then be used by NN to impute the target column.
    
    Args:
        df_with_missing (pd.core.frame.DataFrame): The dataframe with the data.
        feature_names (List[str]): Names of feature columns
        target (str): Name of the target column
        
    Returns:
        imputed_values_with_flag (pd.core.frame.DataFrame): The dataframe with imputed values and flags.
    '''
    pollutants_except_target = [i for i in pollutants_list if i != target]
    
    # Flag the data that was imputed
    imputed_flag = df_with_missing[pollutants_except_target]

    
    warnings.filterwarnings('ignore')
    
    for pollutant in pollutants_except_target:
        # Create the flag column for the pollutant
        imputed_flag[f'{pollutant}_imputed_flag'] = np.where(imputed_flag[pollutant].isnull(), 'interpolated', None)
        imputed_flag.drop(pollutant, axis=1, inplace=True)
        
        # Impute a value to the first one if it is missing, because interpolate does not fix the first value
        if np.any(df_with_missing.loc[[df_with_missing.index[0]], [pollutant]].isnull()):
            df_with_missing.loc[[df_with_missing.index[0]], [pollutant]] = [12]
    
    # Interpolate missing values
    imputed_values = df_with_missing[feature_names].interpolate(method='linear')
    
    imputed_values_with_flag = imputed_values.join(imputed_flag)
    
    return imputed_values_with_flag


def impute_target_missing_values_neural_network(
    df_with_missing: pd.core.frame.DataFrame,
    model: tf.keras.Model,
    scaler: StandardScaler,
    baseline_imputed: pd.core.frame.DataFrame,
    target: str,
) -> pd.core.frame.DataFrame:
    '''
    Imputes data to non-target variables using interpolation.
    This data can then be used by NN to impute the target column.
    
    Args:
        df_with_missing (pd.core.frame.DataFrame): The dataframe with the data.
        model (tf.keras.Model): Model
        scaler (StandardScaler): scaler
        baseline_imputed (pd.core.frame.DataFrame): The dataframe with imputed values and flags for nontarget.
        target (str): Name of the target column
        
    Returns:
        data_with_imputed (pd.core.frame.DataFrame): The dataframe with imputed values and flags.
    '''
    # Metadata columns that we want to output in the end
    metadata_columns = ['DateTime', 'Station', 'Latitude', 'Longitude']
    
    # Save the data and imputed flags of nontarget pollutant for outputting later
    baseline_imputed_data_and_flags = baseline_imputed[[i for i in list(baseline_imputed.columns) if i in pollutants_list or 'flag' in i]]
    
    # Flag the data that will be imputed with NN
    imputed_flag = df_with_missing[[target]]
    imputed_flag[f'{target}_imputed_flag'] = np.where(imputed_flag[target].isnull(), 'neural network', None)
    imputed_flag.drop(target, axis=1, inplace=True)
    
    # For predicting drop the flags, because the neural network doesnt take them
    baseline_imputed = baseline_imputed[[i for i in list(baseline_imputed.columns) if 'flag' not in i]]
    # For predicting we just need the rows where the target pollutant is actually missing
    baseline_imputed = baseline_imputed[df_with_missing[target].isnull()]

    # Predict the target
    baseline_imputed = scaler.transform(baseline_imputed)
    predicted_target = model.predict(baseline_imputed)
    
    # Replace the missing values in the original dataframe with predicted ones
    index_of_missing = df_with_missing[target].isnull()
    data_with_imputed = df_with_missing.copy()
    data_with_imputed.loc[index_of_missing, target] = predicted_target
    
    # Add the flag to the predicted values
    data_with_imputed = data_with_imputed[
        metadata_columns + [target]
    ].join(imputed_flag).join(baseline_imputed_data_and_flags)
    
    # Rearrange the columns so they are in a nicer order for visual representation
    order_of_columns = metadata_columns + pollutants_list + [i + '_imputed_flag' for i in pollutants_list]
    data_with_imputed = data_with_imputed[order_of_columns]
    
    return data_with_imputed
