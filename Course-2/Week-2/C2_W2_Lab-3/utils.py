import ipywidgets as widgets
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
import numpy as np
from sklearn import metrics
import random
from dataclasses import dataclass
from datetime import datetime, timedelta
from ipywidgets import interact
from typing import Callable, List, Tuple, Dict, Optional
import pickle


FONT_SIZE_TICKS = 15
FONT_SIZE_TITLE = 25
FONT_SIZE_AXES = 20


def prepare_data(df: pd.core.frame.DataFrame, turb_id: int) -> pd.core.frame.DataFrame:
    """Pre-process data before feeding to neural networks for training.
    This includes:
    - Resampling to an hourly basis
    - Using data from a single turbine
    - Format datetime
    - Mask abnormal values
    - Re-order columns

    Args:
        df (pandas.core.frame.DataFrame): The curated data from the previous lab.
        turb_id (int): ID of the turbine to use.

    Returns:
        pandas.core.frame.DataFrame: Processed dataframe.
    """
    df = df[5::6]
    df = df[df.TurbID == turb_id]
    df = df.drop(["TurbID"], axis=1)
    df.index = pd.to_datetime(df.pop("Datetime"), format="%Y-%m-%d %H:%M")
    df = df.mask(df.Include == False, -1)
    df = df.drop(["Include"], axis=1)

    df = df[
        [
            "Wspd",
            "Etmp",
            "Itmp",
            "Prtv",
            "WdirCos",
            "WdirSin",
            "NdirCos",
            "NdirSin",
            "PabCos",
            "PabSin",
            "Patv",
        ]
    ]

    return df


def normalize_data(
    train_data: pd.core.frame.DataFrame,
    val_data: pd.core.frame.DataFrame,
    test_data: pd.core.frame.DataFrame,
) -> Tuple[
    np.ndarray, np.ndarray, np.ndarray, pd.core.series.Series, pd.core.series.Series
]:
    """Normalizes train, val and test splits.

    Args:
        train_data (pd.core.frame.DataFrame): Train split.
        val_data (pd.core.frame.DataFrame): Validation split.
        test_data (pd.core.frame.DataFrame): Test split.

    Returns:
        tuple: Normalized splits with training mean and standard deviation.
    """
    train_mean = train_data.mean()
    train_std = train_data.std()

    train_data = (train_data - train_mean) / train_std
    val_data = (val_data - train_mean) / train_std
    test_data = (test_data - train_mean) / train_std

    return train_data, val_data, test_data, train_mean, train_std


@dataclass
class DataSplits:
    """Class to encapsulate normalized/unnormalized train, val, test, splits."""

    train_data: pd.core.frame.DataFrame
    val_data: pd.core.frame.DataFrame
    test_data: pd.core.frame.DataFrame
    train_mean: pd.core.series.Series
    train_std: pd.core.series.Series
    train_df_unnormalized: pd.core.frame.DataFrame
    val_df_unnormalized: pd.core.frame.DataFrame
    test_df_unnormalized: pd.core.frame.DataFrame


def train_val_test_split(df: pd.core.frame.DataFrame) -> DataSplits:
    """Splits a dataframe into train, val and test.

    Args:
        df (pd.core.frame.DataFrame): The data to split.

    Returns:
        data_splits (DataSplits): An instance that encapsulates normalized/unnormalized splits.
    """
    n = len(df)
    train_df = df[0 : int(n * 0.7)]
    val_df = df[int(n * 0.7) : int(n * 0.9)]
    test_df = df[int(n * 0.9) :]

    train_df_un = train_df.copy(deep=True)
    val_df_un = val_df.copy(deep=True)
    test_df_un = test_df.copy(deep=True)

    train_df_un = train_df_un.mask(train_df_un.Patv == -1, np.nan)
    val_df_un = val_df_un.mask(val_df_un.Patv == -1, np.nan)
    test_df_un = test_df_un.mask(test_df_un.Patv == -1, np.nan)

    train_df, val_df, test_df, train_mn, train_st = normalize_data(
        train_df, val_df, test_df
    )

    ds = DataSplits(
        train_data=train_df,
        val_data=val_df,
        test_data=test_df,
        train_mean=train_mn,
        train_std=train_st,
        train_df_unnormalized=train_df_un,
        val_df_unnormalized=val_df_un,
        test_df_unnormalized=test_df_un,
    )

    return ds


def plot_time_series(data_splits: DataSplits) -> None:
    """Plots time series of active power vs the other features.

    Args:
        data_splits (DataSplits): Turbine data.
    """
    train_df, val_df, test_df = (
        data_splits.train_df_unnormalized,
        data_splits.val_df_unnormalized,
        data_splits.test_df_unnormalized,
    )

    def plot_time_series(feature):
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(20, 12))
        ax1.plot(train_df["Patv"], color="blue", label="training")
        ax1.plot(val_df["Patv"], color="green", label="validation")
        ax1.plot(test_df["Patv"], color="red", label="test")
        ax1.set_title("Time series of Patv (target)", fontsize=FONT_SIZE_TITLE)
        ax1.set_ylabel("Active Power (kW)", fontsize=FONT_SIZE_AXES)
        ax1.set_xlabel("Date", fontsize=FONT_SIZE_AXES)
        ax1.legend(fontsize=15)
        ax1.tick_params(axis="both", labelsize=FONT_SIZE_TICKS)

        ax2.plot(train_df[feature], color="blue", label="training")
        ax2.plot(val_df[feature], color="green", label="validation")
        ax2.plot(test_df[feature], color="red", label="test")
        ax2.set_title(f"Time series of {feature} (predictor)", fontsize=FONT_SIZE_TITLE)
        ax2.set_ylabel(f"{feature}", fontsize=FONT_SIZE_AXES)
        ax2.set_xlabel("Date", fontsize=FONT_SIZE_AXES)
        ax2.legend(fontsize=15)
        ax2.tick_params(axis="both", labelsize=FONT_SIZE_TICKS)

        plt.tight_layout()
        plt.show()

    feature_selection = widgets.Dropdown(
        options=[f for f in list(train_df.columns) if f != "Patv"],
        description="Feature",
    )

    interact(plot_time_series, feature=feature_selection)


def compute_metrics(
    true_series: np.ndarray, forecast: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Computes MSE and MAE between two time series.

    Args:
        true_series (np.ndarray): True values.
        forecast (np.ndarray): Forecasts.

    Returns:
        tuple: MSE and MAE metrics.
    """

    mse = tf.keras.metrics.mean_squared_error(true_series, forecast).numpy()
    mae = tf.keras.metrics.mean_absolute_error(true_series, forecast).numpy()

    return mse, mae


class WindowGenerator:
    """Class that handles all of the windowing and plotting logic for time series."""

    def __init__(
        self,
        input_width,
        label_width,
        shift,
        train_df,
        val_df,
        test_df,
        label_columns=["Patv"],
    ):
        self.train_df = train_df
        self.val_df = val_df
        self.test_df = test_df

        self.label_columns = label_columns
        if label_columns is not None:
            self.label_columns_indices = {
                name: i for i, name in enumerate(label_columns)
            }
        self.column_indices = {name: i for i, name in enumerate(train_df.columns)}

        self.input_width = input_width
        self.label_width = label_width
        self.shift = shift

        self.total_window_size = input_width + shift

        self.input_slice = slice(0, input_width)
        self.input_indices = np.arange(self.total_window_size)[self.input_slice]

        self.label_start = self.total_window_size - self.label_width
        self.labels_slice = slice(self.label_start, None)
        self.label_indices = np.arange(self.total_window_size)[self.labels_slice]

    def split_window(self, features):
        inputs = features[:, self.input_slice, :]
        labels = features[:, self.labels_slice, :]
        if self.label_columns is not None:
            labels = tf.stack(
                [
                    labels[:, :, self.column_indices[name]]
                    for name in self.label_columns
                ],
                axis=-1,
            )

        inputs.set_shape([None, self.input_width, None])
        labels.set_shape([None, self.label_width, None])
        return inputs, labels

    def plot(self, model=None, plot_col="Patv", max_subplots=1):
        inputs, labels = self.example
        plt.figure(figsize=(20, 6))
        plot_col_index = self.column_indices[plot_col]
        max_n = min(max_subplots, len(inputs))
        for n in range(max_n):
            plt.subplot(max_n, 1, n + 1)
            plt.title("Inputs (past) vs Labels (future predictions)", fontsize=FONT_SIZE_TITLE)
            plt.ylabel(f"{plot_col} (normalized)", fontsize=FONT_SIZE_AXES)
            plt.plot(
                self.input_indices,
                inputs[n, :, plot_col_index],
                color="green",
                linestyle="--",
                label="Inputs",
                marker="o",
                markersize=10,
                zorder=-10,
            )

            if self.label_columns:
                label_col_index = self.label_columns_indices.get(plot_col, None)
            else:
                label_col_index = plot_col_index

            if label_col_index is None:
                continue

            plt.plot(
                self.label_indices,
                labels[n, :, label_col_index],
                color="orange",
                linestyle="--",
                label="Labels",
                markersize=10,
                marker="o"
            )
            if model is not None:
                predictions = model(inputs)
                plt.scatter(
                    self.label_indices,
                    predictions[n, :, label_col_index],
                    marker="*",
                    edgecolors="k",
                    label="Predictions",
                    c="pink",
                    s=64,
                )
            plt.legend(fontsize=FONT_SIZE_TICKS)
        plt.tick_params(axis="both", labelsize=FONT_SIZE_TICKS)
        plt.xlabel("Timestep", fontsize=FONT_SIZE_AXES)

    def plot_long(
        self,
        model,
        data_splits,
        plot_col="Patv",
        time_steps_future=1,
        baseline_mae=None,
    ):
        train_mean, train_std = data_splits.train_mean, data_splits.train_std
        self.test_size = len(self.test_df)
        self.test_data = self.make_test_dataset(self.test_df, self.test_size)

        inputs, labels = next(iter(self.test_data))

        plt.figure(figsize=(20, 6))
        plot_col_index = self.column_indices[plot_col]

        plt.ylabel(f"{plot_col} (kW)", fontsize=FONT_SIZE_AXES)

        if self.label_columns:
            label_col_index = self.label_columns_indices.get(plot_col, None)
        else:
            label_col_index = plot_col_index

        labels = (labels * train_std.Patv) + train_mean.Patv

        upper = 24 - (time_steps_future - 1)
        lower = self.label_indices[-1] - upper
        self.label_indices_long = self.test_df.index[lower:-upper]

        plt.plot(
            self.label_indices_long[:],
            labels[:, time_steps_future - 1, label_col_index][:],
            label="Labels",
            c="green",
        )

        if model is not None:
            predictions = model(inputs)
            predictions = (predictions * train_std.Patv) + train_mean.Patv
            predictions_for_timestep = predictions[
                :, time_steps_future - 1, label_col_index
            ][:]
            predictions_for_timestep = tf.nn.relu(predictions_for_timestep).numpy()
            plt.plot(
                self.label_indices_long[:],
                predictions_for_timestep,
                label="Predictions",
                c="orange",
                linewidth=3,
            )
            plt.legend(fontsize=FONT_SIZE_TICKS)
            _, mae = compute_metrics(
                labels[:, time_steps_future - 1, label_col_index][:],
                predictions_for_timestep,
            )

            if baseline_mae is None:
                baseline_mae = mae

            print(
                f"\nMean Absolute Error (kW): {mae:.2f} for forecast.\n\nImprovement over random baseline: {100*((baseline_mae -mae)/baseline_mae):.2f}%"
            )
        plt.title("Predictions vs Real Values for Test Split", fontsize=FONT_SIZE_TITLE)
        plt.xlabel("Date", fontsize=FONT_SIZE_AXES)
        plt.tick_params(axis="both", labelsize=FONT_SIZE_TICKS)
        return mae

    def make_test_dataset(self, data, bs):
        data = np.array(data, dtype=np.float32)
        ds = tf.keras.utils.timeseries_dataset_from_array(
            data=data,
            targets=None,
            sequence_length=self.total_window_size,
            sequence_stride=1,
            shuffle=False,
            batch_size=bs,
        )

        ds = ds.map(self.split_window)

        return ds

    def make_dataset(self, data):
        data = np.array(data, dtype=np.float32)
        ds = tf.keras.utils.timeseries_dataset_from_array(
            data=data,
            targets=None,
            sequence_length=self.total_window_size,
            sequence_stride=1,
            shuffle=True,
            batch_size=32,
        )

        ds = ds.map(self.split_window)

        return ds

    @property
    def train(self):
        return self.make_dataset(self.train_df)

    @property
    def val(self):
        return self.make_dataset(self.val_df)

    @property
    def test(self):
        return self.make_dataset(self.test_df)

    @property
    def example(self):
        result = getattr(self, "_example", None)
        if result is None:
            result = next(iter(self.train))
            self._example = result
        return result


def generate_window(
    train_df: pd.core.frame.DataFrame,
    val_df: pd.core.frame.DataFrame,
    test_df: pd.core.frame.DataFrame,
    days_in_past: int,
    width: int = 24
) -> WindowGenerator:
    """Creates a windowed dataset given the train, val, test splits and the number of days into the past.

    Args:
        train_df (pd.core.frame.DataFrame): Train split.
        val_df (pd.core.frame.DataFrame): Val Split.
        test_df (pd.core.frame.DataFrame): Test split.
        days_in_past (int): How many days into the past will be used to predict the next 24 hours.

    Returns:
        WindowGenerator: The windowed dataset.
    """
    OUT_STEPS = 24
    multi_window = WindowGenerator(
        input_width=width * days_in_past,
        label_width=OUT_STEPS,
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        shift=OUT_STEPS,
    )
    return multi_window


def create_model(num_features: int, days_in_past: int) -> tf.keras.Model:
    """Creates a Conv-LSTM model for time series prediction.

    Args:
        num_features (int): Number of features used for prediction.
        days_in_past (int): Number of days into the past to predict next 24 hours.

    Returns:
        tf.keras.Model: The uncompiled model.
    """
    CONV_WIDTH = 3
    OUT_STEPS = 24
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Masking(
                mask_value=-1.0, input_shape=(days_in_past * 24, num_features)
            ),
            tf.keras.layers.Lambda(lambda x: x[:, -CONV_WIDTH:, :]),
            tf.keras.layers.Conv1D(256, activation="relu", kernel_size=(CONV_WIDTH)),
            tf.keras.layers.Bidirectional(
                tf.keras.layers.LSTM(32, return_sequences=True)
            ),
            tf.keras.layers.Bidirectional(
                tf.keras.layers.LSTM(32, return_sequences=False)
            ),
            tf.keras.layers.Dense(
                OUT_STEPS * 1, kernel_initializer=tf.initializers.zeros()
            ),
            tf.keras.layers.Reshape([OUT_STEPS, 1]),
        ]
    )

    return model


def compile_and_fit(
    model: tf.keras.Model, window: WindowGenerator, patience: int = 2
) -> tf.keras.callbacks.History:
    """Compiles and trains a model given a patience threshold.

    Args:
        model (tf.keras.Model): The model to train.
        window (WindowGenerator): The windowed data.
        patience (int, optional): Patience threshold to stop training. Defaults to 2.

    Returns:
        tf.keras.callbacks.History: The training history.
    """
    EPOCHS = 20
    
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=patience, mode="min"
    )

    model.compile(
        loss=tf.keras.losses.MeanSquaredError(),
        optimizer=tf.keras.optimizers.Adam(),
    )

    tf.random.set_seed(432)
    np.random.seed(432)
    random.seed(432)

    history = model.fit(
        window.train, epochs=EPOCHS, validation_data=window.val, callbacks=[early_stopping]
    )
    
    if len(history.epoch) < EPOCHS:
        print("\nTraining stopped early to prevent overfitting, as the validation loss is increasing for two consecutive steps.")
    
    return history


def train_conv_lstm_model(
    data: pd.core.frame.DataFrame, features: List[str], days_in_past: int
) -> Tuple[WindowGenerator, tf.keras.Model, DataSplits]:
    """Trains the Conv-LSTM model for time series prediction.

    Args:
        data (pd.core.frame.DataFrame): The dataframe to be used.
        data (list[str]): The features to use for forecasting.
        days_in_past (int): How many days in the past to use to forecast the next 24 hours.

    Returns:
        tuple: The windowed dataset, the model that handles the forecasting logic and the data used.
    """
    data_splits = train_val_test_split(data[features])

    train_data, val_data, test_data, train_mean, train_std = (
        data_splits.train_data,
        data_splits.val_data,
        data_splits.test_data,
        data_splits.train_mean,
        data_splits.train_std,
    )

    window = generate_window(train_data, val_data, test_data, days_in_past)
    num_features = window.train_df.shape[1]

    model = create_model(num_features, days_in_past)
    history = compile_and_fit(model, window)
    
    return window, model, data_splits


def prediction_plot(
    func: Callable, model: tf.keras.Model, data_splits: DataSplits, baseline_mae: float
) -> None:
    """Plot an interactive visualization of predictions vs true values.

    Args:
        func (Callable): Function to close over. Should be the plot_long method from the WindowGenerator instance.
        model (tf.keras.Model): The trained model.
        data_splits (DataSplits): The data used.
        baseline_mae (float): MAE of baseline to compare against.
    """

    def _plot(time_steps_future):
        mae = func(
            model,
            data_splits,
            time_steps_future=time_steps_future,
            baseline_mae=baseline_mae,
        )

    time_steps_future_selection = widgets.IntSlider(
        value=24,
        min=1,
        max=24,
        step=1,
        description="Hours into future",
        disabled=False,
        continuous_update=False,
        orientation="horizontal",
        readout=True,
        readout_format="d",
        layout={"width": "500px"},
        style={"description_width": "initial"},
    )

    interact(_plot, time_steps_future=time_steps_future_selection)
    
    
def random_forecast(
    data_splits: DataSplits, n_days: int = 1
) -> Tuple[WindowGenerator, tf.keras.Model]:
    """Generates a random forecast for a time window.

    Args:
        data_splits (DataSplits): The data to be used.
        n_days (int, optional): Period from which to draw the random values. Defaults to 1.

    Returns:
        tuple: The windowed dataset and the model that handles the forecasting logic.
    """
    train_data, val_data, test_data = (
        data_splits.train_data,
        data_splits.val_data,
        data_splits.test_data,
    )

    random_window = generate_window(train_data, val_data, test_data, n_days)

    class randomBaseline(tf.keras.Model):
        def call(self, inputs):
            tf.random.set_seed(424)
            np.random.seed(424)
            random.seed(424)
            stacked = tf.random.shuffle(inputs)

            return stacked[:, :, -1:]

    random_baseline = randomBaseline()
    random_baseline.compile(
        loss=tf.keras.losses.MeanSquaredError(),
        metrics=[tf.keras.metrics.MeanAbsoluteError()],
    )

    return random_window, random_baseline


def repeat_forecast(
    data_splits: DataSplits, shift: int=24
) -> Tuple[WindowGenerator, tf.keras.Model]:
    """Performs a repeated forecast logic.

    Args:
        data_splits (DataSplits): The data to be used.
        n_days (int): Period to repeat.

    Returns:
        tuple: The windowed dataset and the model that handles the forecasting logic.
    """
    train_data, val_data, test_data = (
        data_splits.train_data,
        data_splits.val_data,
        data_splits.test_data,
    )
    repeat_window = generate_window(train_data, val_data, test_data, 1, shift)

    class RepeatBaseline(tf.keras.Model):
        def call(self, inputs):
            return inputs[:, :, -1:]

    repeat_baseline = RepeatBaseline()
    repeat_baseline.compile(
        loss=tf.keras.losses.MeanSquaredError(),
        metrics=[tf.keras.metrics.MeanAbsoluteError()],
    )

    return repeat_window, repeat_baseline


def interact_repeat_forecast(
    data_splits: DataSplits, baseline_mae: float
) -> None:
    """Plot an interactive visualization of predictions vs true values.

    Args:
        func (Callable): Function to close over. Should be the plot_long method from the WindowGenerator instance.
        model (tf.keras.Model): The trained model.
        data_splits (DataSplits): The data used.
        baseline_mae (float): MAE of baseline to compare against.
    """

    def _plot(shift):
        repeat_window, repeat_baseline = repeat_forecast(data_splits, shift=shift)
        _ = repeat_window.plot_long(repeat_baseline, data_splits, baseline_mae=baseline_mae)

    shift_selection = widgets.IntSlider(
        value=24,
        min=1,
        max=24,
        step=1,
        description="Hours into future",
        disabled=False,
        continuous_update=False,
        orientation="horizontal",
        readout=True,
        readout_format="d",
        layout={"width": "500px"},
        style={"description_width": "initial"},
    )

    interact(_plot, shift=shift_selection)

def moving_avg_forecast(data_splits: DataSplits, n_days: int) -> Tuple[WindowGenerator, tf.keras.Model]:
    """Performs a moving average forecast logic.

    Args:
        data_splits (DataSplits): The data to be used.
        n_days (int): Period to repeat.

    Returns:
        tuple: The windowed dataset and the model that handles the forecasting logic.
    """
    train_data, val_data, test_data = (
        data_splits.train_data,
        data_splits.val_data,
        data_splits.test_data,
    )
    moving_avg_window = generate_window(train_data, val_data, test_data, n_days)

    class avgBaseline(tf.keras.Model):
        def call(self, inputs):
            m = tf.math.reduce_mean(inputs, axis=1)
            stacked = tf.stack([m for _ in range(inputs.shape[1])], axis=1)

            return stacked[:, :, -1:]

    moving_avg_baseline = avgBaseline()
    moving_avg_baseline.compile(
        loss=tf.keras.losses.MeanSquaredError(),
        metrics=[tf.keras.metrics.MeanAbsoluteError()],
    )

    return moving_avg_window, moving_avg_baseline


def add_wind_speed_forecasts(
    df: pd.core.frame.DataFrame, add_noise=False
) -> pd.core.frame.DataFrame:
    """Creates syntethic wind speed forecasts. The more into the future, the more noise these have.

    Args:
        df (pd.core.frame.DataFrame): Dataframe with data from turbine.
        periods (list, optional): Periods for which to create the forecast. Defaults to [*range(1, 30, 1)].

    Returns:
        pd.core.frame.DataFrame: The new dataframe with the synth forecasts.
    """

    df_2 = df.copy(deep=True)
    # Periods for which to create the forecast.
    periods=[*range(1, 30, 1)]
    
    for period in periods:
        
        if add_noise == "linearly_increasing":
            np.random.seed(8752)
            noise_level = 0.2 * period
            noise = np.random.randn(len(df)) * noise_level
        
        elif add_noise == "mimic_real_forecast":
            np.random.seed(8752)
            noise_level = 2 + 0.05 * period
            noise = np.random.randn(len(df)) * noise_level
        else:
            noise = 0
        
        padding_slice = df_2["Wspd"][-period:].to_numpy()
        values = np.concatenate((df_2["Wspd"][period:].values, padding_slice)) + noise
        
        df_2[f"fc-{period}h"] = values

    return df_2


def plot_forecast_with_noise(
    data_with_wspd_forecasts: pd.core.frame.DataFrame,
) -> None:
    """Creates an interactive plot that shows how the synthetic forecasts change when the future horizon is changed.

    Args:
        data_with_wspd_forecasts (pd.core.frame.DataFrame): Dataframe that includes synth forecasts.
    """

    def _plot(noise_level):
        fig, ax = plt.subplots(figsize=(20, 6))

        df = data_with_wspd_forecasts
        synth_data = df[f"fc-{noise_level}h"][
            5241 - noise_level : -noise_level
        ].values
        synth_data = tf.nn.relu(synth_data).numpy()
        real_data = df["Wspd"][5241:].values
        real_data = tf.nn.relu(real_data).numpy()

        mae = metrics.mean_absolute_error(real_data, synth_data)

        print(f"\nMean Absolute Error (m/s): {mae:.2f} for forecast\n")
        ax.plot(df.index[5241:], real_data, label="true values")
        ax.plot(
            df.index[5241:],
            synth_data,
            label="syntethic predictions",
        )

        ax.set_title("Generated wind speed forecasts", fontsize=FONT_SIZE_TITLE)
        ax.set_ylabel("Wind Speed (m/s)", fontsize=FONT_SIZE_AXES)
        ax.set_xlabel("Date", fontsize=FONT_SIZE_AXES)
        ax.tick_params(axis="both", labelsize=FONT_SIZE_TICKS)
        ax.legend()

    noise_level_selection = widgets.IntSlider(
        value=1,
        min=1,
        max=25,
        step=1,
        description="Noise level in m/s (low to high)",
        disabled=False,
        continuous_update=False,
        orientation="horizontal",
        readout=False,
        layout={"width": "500px"},
        style={"description_width": "initial"},
    )

    interact(_plot, noise_level=noise_level_selection)


def window_plot(data_splits: DataSplits) -> None:
    """Creates an interactive plots to show how the data is windowed depending on the number of days into the past that are used to forecast the next 24 hours.

    Args:
        data_splits (DataSplits): Data used.
    """
    train_data, val_data, test_data = (
        data_splits.train_data,
        data_splits.val_data,
        data_splits.test_data,
    )

    def _plot(time_steps_past):
        window = generate_window(train_data, val_data, test_data, time_steps_past)
        window.plot()

    time_steps_past_selection = widgets.IntSlider(
        value=1,
        min=1,
        max=14,
        step=1,
        description="Days before",
        disabled=False,
        continuous_update=False,
        orientation="horizontal",
        readout=True,
        readout_format="d",
        layout={"width": "500px"},
        style={"description_width": "initial"},
    )

    interact(_plot, time_steps_past=time_steps_past_selection)

    
def load_weather_forecast() -> Dict[str, Dict[List[datetime], List[float]]]:
    """Loads the wind data and forecast for three locations and returns it in a form of dictionary.
    """
    with open("data/weather_forecast.pkl", "rb") as f:
        weather_forecasts = pickle.load(f)
    return weather_forecasts


def plot_forecast(weather_forecasts: Dict[str, Dict[List[datetime], List[float]]]) -> None:
    """Creates an interactive plot of true values vs forecasts for the wind data.

    Args:
        weather_forecasts (dict): History of weather and weather forecasts.
    """
    def _plot(city, time_steps_future):
        format_timestamp = "%Y-%m-%d %H:%M:%S"

        weather_forecast = weather_forecasts[city]
        
        dates_real, winds_real = weather_forecast[0]
        dates_real = [datetime.strptime(i, format_timestamp) for i in dates_real]
        dates_forecast, winds_forecast = weather_forecast[time_steps_future]
        dates_forecast = [datetime.strptime(i, format_timestamp) for i in dates_forecast]

        # Set the min and max date for plotting, so it always plots the same
        min_date = datetime.strptime("2022-11-16 18:00:00", format_timestamp)
        max_date = datetime.strptime("2023-01-11 15:00:00", format_timestamp)
        
        # Find the overlap of the data and limit it to the plotting range
        dates_real, dates_forecast, winds_real, winds_forecast = prepare_wind_data(
            dates_real, dates_forecast, winds_real, winds_forecast, min_date, max_date
        )
        
        fig, ax = plt.subplots(figsize=(20, 6))
        ax.plot(dates_real, winds_real, label="Actual windspeed")
        ax.plot(dates_forecast, winds_forecast, label=f"Forecasted windspeed {time_steps_future} Hours in the Future")
        ax.set_title(f"History of Actual vs Forecasted Windspeed in {city}", fontsize=25)
        ax.set_ylabel("Wind Speed (m/s)", fontsize=20)
        ax.set_xlabel("Date", fontsize=20)
        ax.tick_params(axis="both", labelsize=15)
        ax.legend(fontsize=15)
        
        mae = metrics.mean_absolute_error(winds_real, winds_forecast)
        print(f"\nMean Absolute Error (m/s): {mae:.2f} for forecast\n")
       
    city_selection = widgets.Dropdown(
        options=weather_forecasts.keys(),
        description='City',
    )
    time_steps_future_selection = widgets.IntSlider(
        value=1,
        min=3,
        max=120,
        step=3,
        description="Hours into future",
        disabled=False,
        continuous_update=False,
        orientation="horizontal",
        readout=True,
        readout_format="d",
        layout={"width": "500px"},
        style={"description_width": "initial"},
    )

    interact(_plot, city=city_selection, time_steps_future=time_steps_future_selection)

    
def prepare_wind_data(
    dates0: List[datetime],
    dates1: List[datetime],
    winds0: List[float],
    winds1: List[float],
    min_bound: Optional[str]=None,
    max_bound: Optional[str]=None
) -> Tuple[List[datetime], List[datetime], List[float], List[float]]:
    """Takes in two datasets of wind data.
    Finds the data points that appear in both datasets (at the same time) and are between the specified time bounds.

    Args:
        dates0 (list): list of dates for the first dataset
        dates1 (list): list of dates for the second dataset
        winds0 (list): list of wind speed for the first dataset (corresponding to dates0)
        winds1 (list): list of wind speed for the second dataset (corresponding to dates1)
        min_bound (datetime): minimum bound for plotting
        max_bound (datetime): maximum bound for plotting
    """
    winds0_overlap = []
    winds1_overlap = []
    dates0_overlap = []
    dates1_overlap = []
    
    # Only take the dates that are in both datasets and within the limits if specified
    for date, wind in zip(dates0, winds0):
        if (date in dates1 and 
            (min_bound is None or date > min_bound) and
            (max_bound is None or date < max_bound)
           ):
            winds0_overlap.append(wind)
            dates0_overlap.append(date)
    for date, wind in zip(dates1, winds1):
        if (date in dates0 and 
            (min_bound is None or date > min_bound) and
            (max_bound is None or date < max_bound)
           ):
            winds1_overlap.append(wind)
            dates1_overlap.append(date)
    
    return dates0_overlap, dates1_overlap, winds0_overlap, winds1_overlap


def plot_mae_forecast(weather_forecasts: Dict[str, Dict[List[datetime], List[float]]]) -> None:
    """Creates an interactive plot MAE of wind forecasts.

    Args:
        weather_forecasts (dict): Weather and weather forecast data.
    """
    def _plot(city):
        weather_forecast = weather_forecasts[city]
        
        times = sorted(weather_forecast.keys())[1::]
        maes = []
        
        dates_real, winds_real = weather_forecast[0]
        for time in times:
            dates_forecast, winds_forecast = weather_forecast[time]
            dates_real, dates_forecast, winds_real, winds_forecast = prepare_wind_data(
                dates_real, dates_forecast, winds_real, winds_forecast
            )
            mae = metrics.mean_absolute_error(winds_real, winds_forecast)
            maes.append(mae)
            
        fig, ax = plt.subplots(figsize=(20, 6))
        ax.plot(times, maes, marker="*")
        ax.set_title("Mean Absolute Error of Actual vs Predicted Wind Speed", fontsize=FONT_SIZE_TITLE)
        ax.set_ylabel("Mean Absolute Error (m/s)", fontsize=FONT_SIZE_AXES)
        ax.set_xlabel("Hours into the future", fontsize=FONT_SIZE_AXES)
        ax.tick_params(axis="both", labelsize=FONT_SIZE_TICKS)
               
    city_selection = widgets.Dropdown(
        options=weather_forecasts.keys(),
        description='City',
    )
    
    interact(_plot, city=city_selection)