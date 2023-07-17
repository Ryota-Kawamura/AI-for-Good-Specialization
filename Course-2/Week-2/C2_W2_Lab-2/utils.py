import numpy as np
import pandas as pd
import seaborn as sns
import ipywidgets as widgets
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import shap
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.inspection import permutation_importance
from datetime import datetime, timedelta
from ipywidgets import interact, interact_manual, fixed
from typing import List, Iterable


FONT_SIZE_TICKS = 15
FONT_SIZE_TITLE = 20
FONT_SIZE_AXES = 20


def fix_temperatures(data: pd.core.frame.DataFrame) -> pd.core.frame.DataFrame:
    """Replaces very low temperature values with linear interpolation.

    Args:
        data (pd.core.frame.DataFrame): The dataset.

    Returns:
        pd.core.frame.DataFrame: Dataset with fixed temperatures.
    """
    min_etemp = data["Etmp"].quantile(0.01)
    data["Etmp"] = data["Etmp"].apply(lambda x: np.nan if x < min_etemp else x)
    data["Etmp"] = data["Etmp"].interpolate()
    min_itemp = data["Itmp"].quantile(0.01)
    data["Itmp"] = data["Itmp"].apply(lambda x: np.nan if x < min_itemp else x)
    data["Itmp"] = data["Itmp"].interpolate()

    return data


def tag_abnormal_values(
    df: pd.core.frame.DataFrame, condition: pd.core.series.Series
) -> pd.core.frame.DataFrame:
    """Determines if a given record is an abnormal value.

    Args:
        df (pd.core.frame.DataFrame): The dataset used.
        condition (pd.core.series.Series): Series that includes if a record meets one of the conditions for being an abnormal value.

    Returns:
        pd.core.frame.DataFrame: Dataset with tagger abnormal values.
    """
    indexes = df[condition].index
    df.loc[indexes, "Include"] = False
    return df


def cut_pab_features(raw_data: pd.core.frame.DataFrame) -> pd.core.frame.DataFrame:
    """Deletes redundant Pab features from dataset.

    Args:
        raw_data (pd.core.frame.DataFrame): The dataset used.

    Returns:
        pd.core.frame.DataFrame: The dataset without the redundant Pab features.
    """

    raw_data = raw_data.drop(["Pab2", "Pab3"], axis=1)
    raw_data = raw_data.rename(columns={"Pab1": "Pab"})

    return raw_data


def generate_time_signals(raw_data: pd.core.frame.DataFrame) -> pd.core.frame.DataFrame:
    """Creates time signal features (time-of-day) for the data.

    Args:
        raw_data (pd.core.frame.DataFrame): The dataset uded.

    Returns:
        pd.core.frame.DataFrame: The dataset with the new features.
    """
    if "Day sin" in raw_data.columns:
        return raw_data

    date_time = pd.to_datetime(raw_data.Datetime, format="%Y-%m-%d %H:%M")
    timestamp_s = date_time.map(pd.Timestamp.timestamp)

    day = 24 * 60 * 60

    raw_data["Time-of-day sin"] = np.sin(timestamp_s * (2 * np.pi / day))
    raw_data["Time-of-day cos"] = np.cos(timestamp_s * (2 * np.pi / day))

    return raw_data


def top_n_turbines(
    raw_data: pd.core.frame.DataFrame, n: int
) -> pd.core.frame.DataFrame:
    """Keeps only the top n turbines that produced more energy on average.

    Args:
        raw_data (pd.core.frame.DataFrame): The full dataset.
        n (int): Desired number of turbines to keep.

    Returns:
        pd.core.frame.DataFrame: The dataset with only the data from the top n turbines.
    """
    sorted_patv_by_turbine = (
        raw_data.groupby("TurbID").mean()["Patv"].sort_values(ascending=False)
    )

    top_turbines = list(sorted_patv_by_turbine.index)[:n]

    print(
        f"Original data has {len(raw_data)} rows from {len(raw_data.TurbID.unique())} turbines.\n"
    )

    raw_data = raw_data[raw_data["TurbID"].isin(top_turbines)]

    print(
        f"Sliced data has {len(raw_data)} rows from {len(raw_data.TurbID.unique())} turbines."
    )

    return raw_data


def format_datetime(
    data: pd.core.frame.DataFrame, initial_date_str: str
) -> pd.core.frame.DataFrame:
    """Formats Day and Tmstamp features into a Datetime feature.

    Args:
        data (pd.core.frame.DataFrame): The original dataset.
        initial_date_str (str): The initial date.

    Returns:
        pd.core.frame.DataFrame: The dataframe with formatted datetime.
    """
    if "Datetime" in data.columns:

        return data

    initial_date = datetime.strptime(initial_date_str, "%d %m %Y").date()

    data["Date"] = data.apply(
        lambda x: str(initial_date + timedelta(days=(x.Day - 1))), axis=1
    )

    data["Datetime"] = data.apply(
        lambda x: datetime.strptime(f"{x.Date} {x.Tmstamp}", "%Y-%m-%d %H:%M"), axis=1
    )

    data.drop(["Day", "Tmstamp", "Date"], axis=1, inplace=True)

    data = data[["Datetime"] + [col for col in list(data.columns) if col != "Datetime"]]

    return data


def transform_angles(
    data: pd.core.frame.DataFrame, feature: str, drop_original: bool = True
):
    """Transform angles into their Sin/Cos encoding.

    Args:
        data (pd.core.frame.DataFrame): The dataset used.
        feature (str): Name of the angle feature.
        drop_original (bool, optional): Wheter to drop the original column from the dataset. Defaults to True.
    """
    # np.cos and np.sin expect angles in radians
    rads = data[feature] * np.pi / 180

    # Compute Cos and Sin
    data[f"{feature}Cos"] = np.cos(rads)
    data[f"{feature}Sin"] = np.sin(rads)

    if drop_original:
        data.drop(feature, axis=1, inplace=True)


def plot_wind_speed_vs_power(
    ax: plt.Axes,
    x1: Iterable,
    y1: Iterable,
    x2: Iterable,
    y2: Iterable
):
    """Plots wind speed on x-axis and wind power on y axis.

    Args:
        ax (mpl.axes._subplots.AxesSubplot): Axis on which to plot.
        x1, y1: The x, y original data to be plotted. Both can be None if not available.
        x2, y2: The x, y data model to be plotted. Both can be None if not available.
    """
    # Plot the original data
    ax.scatter(
        x1, y1, color="blue", edgecolors="white", s=15, label="actual"
    )
    # Plot the model
    ax.scatter(
        x2, y2,
        color="orange", edgecolors="black", s=15, marker="D", label="model"
    )
    ax.set_xlabel("Wind Speed (m/s)", fontsize=FONT_SIZE_AXES)
    ax.set_ylabel("Active Power (kW)", fontsize=FONT_SIZE_AXES)
    ax.set_title("Wind Speed vs. Power Output", fontsize=FONT_SIZE_TITLE)
    ax.tick_params(labelsize=FONT_SIZE_TICKS)
    ax.legend(fontsize=FONT_SIZE_TICKS)


def plot_predicted_vs_real(
    ax: plt.Axes,
    x1: Iterable,
    y1: Iterable,
    x2: Iterable,
    y2: Iterable
):
    """Plots predicted vs. actual data.

    Args:
        ax (mpl.axes._subplots.AxesSubplot): Axis on which to plot.
        x1, y1: The x, y original data to be plotted. Both can be None if not available.
        x2, y2: The x, y data to plot a line. Both can be None if not available.
    """
    # Plot predicted vs real y
    ax.scatter(
        x1, y1, color="orange", edgecolors="black", label="Predicted vs. actual values", marker="D"
    )
    # Plot straight line
    ax.plot(
        x2, y2, color="blue", linestyle="--", linewidth=4, label="actual = predicted",
    )
    ax.set_xlabel("Actual Power Values (kW)", fontsize=FONT_SIZE_AXES)
    ax.set_ylabel("Predicted Power Values (kW)", fontsize=FONT_SIZE_AXES)
    ax.set_title("Predicted vs. Actual Power Values (kW)", fontsize=FONT_SIZE_TITLE)
    ax.tick_params(labelsize=FONT_SIZE_TICKS)
    ax.legend(fontsize=FONT_SIZE_TICKS)

    
def fit_and_plot_linear_model(data_og: pd.core.frame.DataFrame, turbine: int, features: List[str]):
    # Get the data for the selected turbine
    data = data_og[data_og.TurbID == turbine]

    # Create the linear regression model
    features = list(features)
    y = data["Patv"]
    X = data[features]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    reg = LinearRegression().fit(X_train, y_train)

    # Prepare the data for plotting
    X_plot = data["Wspd"]
    Y_real = data["Patv"]
    y_test_preds = reg.predict(X_test)

    X_eq_Y = np.linspace(0, max([max(y_test), max(y_test_preds)]), 100)

    # Create two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    # Plotting on the left side plot
    if "Wspd" in features:
        plot_wind_speed_vs_power(ax1, X_plot, Y_real, X_test["Wspd"], y_test_preds)
    else:
        plot_wind_speed_vs_power(ax1, X_plot, Y_real, None, None)
        print("The model could not be plotted here as Wspd is not among the features")
    # Plotting on the right side plot
    plot_predicted_vs_real(ax2, y_test, y_test_preds, X_eq_Y, X_eq_Y)
    
    plt.tight_layout()
    plt.show()

    # Create a plot of feature imporance if there is more than one feature
    if len(features) > 1:
        # Create data for feature importance
        bunch = permutation_importance(
            reg, X_test, y_test, n_repeats=10, random_state=42
        )
        imp_means = bunch.importances_mean
        ordered_imp_means_args = np.argsort(imp_means)[::-1]

        results = {}
        for i in ordered_imp_means_args:
            name = list(X_test.columns)[i]
            imp_score = imp_means[i]
            results.update({name: [imp_score]})

        results_df = pd.DataFrame.from_dict(results)

        # Create a plot for feature importance
        fig, ax = plt.subplots(figsize=(7.5, 6))
        ax.set_xlabel("Importance Score", fontsize=FONT_SIZE_AXES)
        ax.set_ylabel("Feature", fontsize=FONT_SIZE_AXES)
        ax.set_title("Feature Importance", fontsize=FONT_SIZE_TITLE)
        ax.tick_params(labelsize=FONT_SIZE_TICKS)

        sns.barplot(data=results_df, orient="h", ax=ax, color="deepskyblue", width=0.3)

        plt.show()

    # Print out the mean absolute error
    mae = metrics.mean_absolute_error(y_test, y_test_preds)
    print(f"Turbine {turbine}, Mean Absolute Error (kW): {mae:.2f}\n")

    
def linear_univariate_model(data_og: pd.core.frame.DataFrame):
    """Creates an interactive plot of the univariate linear model for predicting energy output using wind speed as unique predictor.

    Args:
        data_og (pd.core.frame.DataFrame): The dataset used.
    """

    turbine_selection = widgets.Dropdown(
        options=data_og.TurbID.unique(), description="Turbine"
    )

    interact(fit_and_plot_linear_model, data_og=fixed(data_og), turbine=turbine_selection, features=fixed(["Wspd"]))  

    
def linear_multivariate_model(data_og: pd.core.frame.DataFrame, features: List[str]):
    """Creates an interactive plot to showcase multivariate linear regression.

    Args:
        data_og (pd.core.frame.DataFrame): The data used.
        features (List[str]): List of features to include in the prediction.
    """

    turbine_selection = widgets.Dropdown(
        options=data_og.TurbID.unique(), description="Turbine"
    )

    feature_selection = widgets.SelectMultiple(
        options=features,
        value=list(features),
        description="Features",
        disabled=False,
    )

    interact_manual(fit_and_plot_linear_model, data_og=fixed(data_og), turbine=turbine_selection, features=feature_selection)    


def split_and_normalize(data: pd.core.frame.DataFrame, features: List[str]):
    """Generates the train, test splits and normalizes the data.

    Args:
        data (pd.core.frame.DataFrame): The dataset used.
        features (List[str]): Features to include in the prediction process.

    Returns:
        tuple: The normalized train/test splits along with the train mean and standard deviation.
    """

    X = data[features]
    y = data["Patv"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    to_normalize = ["Wspd", "Etmp", "Itmp", "Prtv"]

    f_to_normalize = [feature for feature in features if feature in to_normalize]

    f_not_to_normalize = [
        feature for feature in features if feature not in to_normalize
    ]

    X_train_mean = X_train[f_to_normalize].mean()
    X_train_std = X_train[f_to_normalize].std()

    y_train_mean = y_train.mean()
    y_train_std = y_train.std()

    X_train[f_to_normalize] = (X_train[f_to_normalize] - X_train_mean) / X_train_std
    X_test[f_to_normalize] = (X_test[f_to_normalize] - X_train_mean) / X_train_std

    y_train = (y_train - y_train_mean) / y_train_std
    y_test = (y_test - y_train_mean) / y_train_std

    X_train = X_train.to_numpy()
    X_test = X_test.to_numpy()
    y_train = y_train.to_numpy()
    y_test = y_test.to_numpy()

    X_train = torch.from_numpy(X_train).type(torch.float)
    X_test = torch.from_numpy(X_test).type(torch.float)
    y_train = torch.from_numpy(y_train).type(torch.float).unsqueeze(dim=1)
    y_test = torch.from_numpy(y_test).type(torch.float).unsqueeze(dim=1)

    return (X_train, X_test, y_train, y_test), (
        X_train_mean,
        X_train_std,
        y_train_mean,
        y_train_std,
    )


def batch_data(
    X_train: torch.Tensor,
    X_test: torch.Tensor,
    y_train: torch.Tensor,
    y_test: torch.Tensor,
    batch_size: int,
):
    """Creates batches from the original data.

    Args:
        X_train (torch.Tensor): Train predictors.
        X_test (torch.Tensor): Test predictors.
        y_train (torch.Tensor): Train target.
        y_test (torch.Tensor): Test target.
        batch_size (int): Desired batch size.

    Returns:
        tuple: Train and test DataLoaders.
    """
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size)

    test_dataset = TensorDataset(X_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    return train_loader, test_loader


class RegressorNet(nn.Module):
    """A vanilla feed forward Neural Network with 3 hidden layers."""

    def __init__(self, input_size):
        super().__init__()

        self.fc_layers = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        x = self.fc_layers(x)
        return x


def compile_model(features: List[str]):
    """Compiles the Pytorch network with an appropriate Loss and Optimizer.

    Args:
        features (List[str]): List of predictors to use.

    Returns:
        tuple: The model, loss function and optimizer used.
    """
    model = RegressorNet(input_size=len(features))
    loss_fn = nn.L1Loss()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=0.001)

    return model, loss_fn, optimizer


def train_model(
    model: RegressorNet,
    loss_fn: torch.nn.modules.loss.L1Loss,
    optimizer: torch.optim.Adam,
    train_loader: torch.utils.data.DataLoader,
    test_loader: torch.utils.data.DataLoader,
    epochs: int,
):
    """Trains the neural network.

    Args:
        model (RegressorNet): An instance of the neural network.
        loss_fn (torch.nn.modules.loss.L1Loss): L1 loss (aka as Mean Absolute Error)
        optimizer (torch.optim.Adam): Adam Optimizer
        train_loader (torch.utils.data.DataLoader): The train data
        test_loader (torch.utils.data.DataLoader): The test data
        epochs (int): Desired number of epochs to train

    Returns:
        RegressorNet: The trained model.
    """

    for epoch in range(epochs):

        model.train()

        for batch, (X, y) in enumerate(train_loader):
            # 1. Forward pass
            y_pred = model(X)

            # 2. Calculate loss
            loss = loss_fn(y_pred, y)

            # 3. Zero grad optimizer
            optimizer.zero_grad()

            # 4. Loss backward
            loss.backward()

            # 5. Step the optimizer
            optimizer.step()

        ### Testing
        model.eval()
        with torch.inference_mode():

            for batch, (X, y) in enumerate(test_loader):
                # 1. Forward pass
                test_pred = model(X)

                # 2. Calculate the loss
                test_loss = loss_fn(test_pred, y)

        if epoch % 1 == 0:
            print(
                f"Epoch: {epoch} | Train loss: {loss:.5f} | Test loss: {test_loss:.5f}"
            )

    return model


def plot_feature_importance(
    model: RegressorNet,
    features: List[str],
    train_loader: torch.utils.data.DataLoader,
    test_loader: torch.utils.data.DataLoader,
):
    """Creates a feature importance plot by using SHAP values.

    Args:
        model (RegressorNet): The trained model.
        features (List[str]): List of predictors used.
        train_loader (torch.utils.data.DataLoader): Training data.
        test_loader (torch.utils.data.DataLoader): Testing data.
    """

    x_train_batch, _ = next(iter(train_loader))
    x_test_batch, _ = next(iter(test_loader))

    model.eval()
    e = shap.DeepExplainer(model, x_train_batch)
    shap_values = e.shap_values(x_test_batch)
    
    means = np.mean(np.abs(shap_values), axis=0)
    results = sorted(zip(features, means), key = lambda x: x[1], reverse=True)
    results_df = pd.DataFrame.from_dict({k: [v] for (k, v) in results})

    # Create a plot for feature importance
    fig, ax = plt.subplots(figsize=(7.5, 6))
    ax.set_xlabel("Importance Score", fontsize=FONT_SIZE_AXES)
    ax.set_ylabel("Feature", fontsize=FONT_SIZE_AXES)
    ax.set_title("Feature Importance", fontsize=FONT_SIZE_TITLE)
    ax.tick_params(labelsize=FONT_SIZE_TICKS)
    sns.barplot(data=results_df, orient="h", ax=ax, color="deepskyblue", width=0.3)
    
    return shap_values


def neural_network(data_og: pd.core.frame.DataFrame, features: List[str]):
    """Creates an interactive plot of the prediction process when using a neural network.

    Args:
        data_og (pd.core.frame.DataFrame): The data used.
        features (List[str]): The features to include in the prediction process.
    """

    def fit_nn(turbine, features):
        data = data_og[data_og.TurbID == turbine]
        features = list(features)
        print(f"Features used: {features}\n")
        print(f"Training your Neural Network...\n")

        (X_train, X_test, y_train, y_test), (
            X_train_mean,
            X_train_std,
            y_train_mean,
            y_train_std,
        ) = split_and_normalize(data, features)
        train_loader, test_loader = batch_data(
            X_train, X_test, y_train, y_test, batch_size=32
        )
        model, loss_fn, optimizer = compile_model(features)
        model = train_model(
            model, loss_fn, optimizer, train_loader, test_loader, epochs=5
        )
        print(f"\nResults:")

        y_test_denormalized = (y_test * y_train_std) + y_train_mean
        test_preds = model(X_test).detach().numpy()
        test_preds_denormalized = (test_preds * y_train_std) + y_train_mean
        X_plot = data["Wspd"]
        Y_real = data["Patv"]
        X_eq_Y = np.linspace(0, max(y_test_denormalized), 100)
        
        print(
            f"Mean Absolute Error: {metrics.mean_absolute_error(y_test_denormalized, test_preds_denormalized):.2f}\n"
        )

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        if "Wspd" in features:
            test_preds = model(X_test).detach().numpy()
            test_preds_denormalized = (test_preds * y_train_std) + y_train_mean

            X_test_2 = X_test.detach().numpy()
            X_test_denormalized = (X_test_2[:, 0] * X_train_std[0]) + X_train_mean[0]
            
            plot_wind_speed_vs_power(ax1, X_plot, Y_real, X_test_denormalized, test_preds_denormalized)
        else:
            plot_wind_speed_vs_power(ax1, X_plot, Y_real, None, None)
            print("The model could not be plotted here as Wspd is not among the features")

        plot_predicted_vs_real(ax2, y_test_denormalized, test_preds_denormalized, X_eq_Y, X_eq_Y)

        plt.show()          

        train_loader, test_loader = batch_data(
            X_train, X_test, y_train, y_test, batch_size=128
        )

        plot_feature_importance(model, features, train_loader, test_loader)

    turbine_selection = widgets.Dropdown(
        options=data_og.TurbID.unique(), description="Turbine"
    )
    feature_selection = widgets.SelectMultiple(
        options=features,
        value=list(features),
        description="Features",
        disabled=False,
    )
    interact_manual(fit_nn, turbine=turbine_selection, features=feature_selection)
