import random
import numpy as np
import pandas as pd
import seaborn as sns
import ipywidgets as widgets
from IPython.display import display
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from ipywidgets import interact
from typing import List


FONT_SIZE_TICKS = 15
FONT_SIZE_TITLE = 25
FONT_SIZE_AXES = 20


def plot_turbines(raw_data: pd.core.frame.DataFrame):
    """Plot turbines' relative positions.

    Args:
        raw_data (pd.core.frame.DataFrame): The dataset used.
    """
    turb_locations = pd.read_csv("./data/turb_location.csv")
    turbs = turb_locations[turb_locations.TurbID.isin(raw_data.TurbID.unique())]
    turbs = turbs.reset_index()
    n = list(raw_data.TurbID.unique())

    fig, ax = plt.subplots(figsize=(16, 6))
    ax.set_title("Spatial location of wind turbines")
    ax.scatter(turbs.x, turbs.y, marker="1", s=500, c="green")
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    for i, txt in enumerate(n):
        ax.annotate(txt, (turbs.x[i], turbs.y[i]))


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
        raw_data.groupby("TurbID").mean()["Patv (kW)"].sort_values(ascending=False)
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


def inspect_missing_values(
    mv_df: pd.core.frame.DataFrame, num_samples: int, output: widgets.Output
):
    """Interactive dataframe inspector to visualize missing values.

    Args:
        mv_df (pd.core.frame.DataFrame): Dataframe with missing values.
        num_samples (int): Number of samples to inspect at any given time.
        output (widgets.Output): Output of the widget (this is for visualization purposes)
    """

    def on_button_clicked(b):
        with output:
            output.clear_output()
            random_index = random.sample([*range(len(mv_df))], num_samples)
            display(mv_df.iloc[random_index].head(num_samples))

    return on_button_clicked


def histogram_plot(df: pd.core.frame.DataFrame, features: List[str], bins: int = 16):
    """Create interactive histogram plots.

    Args:
        df (pd.core.frame.DataFrame): The dataset used.
        features (List[str]): List of features to include in the plot.
        bins (int, optional): Number of bins in the histograms. Defaults to 16.
    """

    def _plot(turbine, feature):
        data = df[df.TurbID == turbine]
        plt.figure(figsize=(8, 5))
        x = data[feature].values
        plt.xlabel(f"{feature}", fontsize=FONT_SIZE_AXES)
        sns.histplot(x, bins=bins)
        plt.ylabel(f"Count", fontsize=FONT_SIZE_AXES)
        plt.title(f"Feature: {feature} - Turbine: {turbine}", fontsize=FONT_SIZE_TITLE)
        plt.tick_params(axis="both", labelsize=FONT_SIZE_TICKS)
        plt.show()

    turbine_selection = widgets.Dropdown(
        options=df.TurbID.unique(), value=df.TurbID.unique()[-1], description="Turbine"
    )

    feature_selection = widgets.Dropdown(
        options=features,
        description="Feature",
    )

    interact(_plot, turbine=turbine_selection, feature=feature_selection)


def histogram_comparison_plot(
    df: pd.core.frame.DataFrame, features: List[str], bins: int = 16
):
    """Create interactive histogram plots.

    Args:
        df (pd.core.frame.DataFrame): The dataset used.
        features (List[str]): List of features to include in the plot.
        bins (int, optional): Number of bins in the histograms. Defaults to 16.
    """

    def _plot(turbine1, turbine2, feature):
        data_1 = df[df.TurbID == turbine1]
        data_2 = df[df.TurbID == turbine2]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 5))

        x_1 = data_1[feature].values
        x_2 = data_2[feature].values

        ax1.set_xlabel(f"{feature}", fontsize=FONT_SIZE_AXES)
        ax2.set_xlabel(f"{feature}", fontsize=FONT_SIZE_AXES)

        ax1.set_ylabel(f"Count", fontsize=FONT_SIZE_AXES)
        ax2.set_ylabel(f"Count", fontsize=FONT_SIZE_AXES)

        sns.histplot(x_1, bins=bins, ax=ax1)
        sns.histplot(x_2, bins=bins, ax=ax2, color="green")

        ax1.set_title(f"Turbine: {turbine1}", fontsize=FONT_SIZE_TITLE)
        ax1.tick_params(axis="both", labelsize=FONT_SIZE_TICKS)

        ax2.set_title(f"Turbine: {turbine2}", fontsize=FONT_SIZE_TITLE)
        ax2.tick_params(axis="both", labelsize=FONT_SIZE_TICKS)

        fig.tight_layout()
        fig.show()

    turbine_selection1 = widgets.Dropdown(
        options=df.TurbID.unique(),
        value=df.TurbID.unique()[-2],
        description="Turbine ID",
        style={"description_width": "initial"},
    )

    turbine_selection2 = widgets.Dropdown(
        options=df.TurbID.unique(),
        value=df.TurbID.unique()[-1],
        description="Another Turbine ID",
        style={"description_width": "initial"},
    )

    feature_selection = widgets.Dropdown(
        options=features,
        description="Feature",
    )

    interact(
        _plot,
        turbine1=turbine_selection1,
        turbine2=turbine_selection2,
        feature=feature_selection,
    )


def box_violin_plot(df: pd.core.frame.DataFrame, features: List[str]):
    """Creates interactive violin/box plots for the dataset.

    Args:
        df (pd.core.frame.DataFrame): The data used.
        features (List[str]): List of features to include in the plot.
    """
    labels = df["TurbID"].unique()

    def _plot(feature="Wspd", plot_type="box"):
        plt.figure(figsize=(18, 8))
        scale = "linear"
        plt.yscale(scale)
        if plot_type == "violin":
            sns.violinplot(
                data=df, y=feature, x="TurbID", order=labels, color="seagreen"
            )
        elif plot_type == "box":
            sns.boxplot(data=df, y=feature, x="TurbID", order=labels, color="seagreen")
        plt.title(f"Feature: {feature}", fontsize=FONT_SIZE_TITLE)
        plt.ylabel(f"{feature}", fontsize=FONT_SIZE_AXES)
        plt.xlabel(f"TurbID", fontsize=FONT_SIZE_AXES)
        plt.tick_params(axis="both", labelsize=FONT_SIZE_TICKS)

        plt.show()

    feature_selection = widgets.Dropdown(
        options=features,
        description="Feature",
    )

    plot_type_selection = widgets.Dropdown(
        options=["violin", "box"], description="Plot Type"
    )

    interact(_plot, feature=feature_selection, plot_type=plot_type_selection)


def scatterplot(df: pd.core.frame.DataFrame, features: List[str]):
    """Creates interactive scatterplots of the data.

    Args:
        df (pd.core.frame.DataFrame): The data used.
        features (List[str]): List of features to include in the plot.
    """
    df_clean = df.dropna(inplace=False)

    def _plot(turbine, var_x, var_y):
        plt.figure(figsize=(12, 6))
        df_clean_2 = df_clean[df_clean.TurbID == turbine]
        x = df_clean_2[var_x].values
        y = df_clean_2[var_y].values

        plt.plot(
            x, y,
            marker='o', markersize=3, markerfacecolor='blue', 
            markeredgewidth=0,
            linestyle='', 
            alpha=0.5
        )
        
        
        plt.xlabel(var_x, fontsize=FONT_SIZE_AXES)
        plt.ylabel(var_y, fontsize=FONT_SIZE_AXES)

        plt.title(f"Scatterplot of {var_x} against {var_y}", fontsize=FONT_SIZE_TITLE)
        plt.tick_params(axis="both", labelsize=FONT_SIZE_TICKS)
        plt.show()

    turbine_selection = widgets.Dropdown(
        options=df.TurbID.unique(), value=df.TurbID.unique()[-1], description="Turbine"
    )

    x_var_selection = widgets.Dropdown(options=features, description="X-Axis")

    y_var_selection = widgets.Dropdown(
        options=features, description="Y-Axis", value="Patv (kW)"
    )

    interact(
        _plot,
        turbine=turbine_selection,
        var_x=x_var_selection,
        var_y=y_var_selection,
    )


def correlation_matrix(data: pd.core.frame.DataFrame):
    """Plots correlation matrix for a given dataset.

    Args:
        data (pd.core.frame.DataFrame): The dataset used.
    """
    plt.figure(figsize=(10, 10))
    sns.heatmap(data.corr(), annot=True, cbar=False, cmap="RdBu", vmin=-1, vmax=1)
    plt.title("Correlation Matrix of Features")
    plt.show()


def plot_time_series(df: pd.core.frame.DataFrame, features: List[str]):
    """Creates interactive plots for the time series in the dataset.

    Args:
        df (pd.core.frame.DataFrame): The data used.
        features (List[str]): Features to include in the plot.
    """

    def plot_time_series(turbine, feature, date_range, fix_temps):
        data = df[df.TurbID == turbine]
        if fix_temps:
            min_etemp = data["Etmp (°C)"].quantile(0.01)
            data["Etmp (°C)"] = data["Etmp (°C)"].apply(
                lambda x: np.nan if x < min_etemp else x
            )
            data["Etmp (°C)"] = data["Etmp (°C)"].interpolate()
            min_itemp = data["Itmp (°C)"].quantile(0.01)
            data["Itmp (°C)"] = data["Itmp (°C)"].apply(
                lambda x: np.nan if x < min_itemp else x
            )
            data["Itmp (°C)"] = data["Itmp (°C)"].interpolate()

        data = data[data.Datetime > date_range[0]]
        data = data[data.Datetime < date_range[1]]
        plt.figure(figsize=(15, 5))
        plt.plot(data["Datetime"], data[feature], "-")
        plt.title(f"Time series of {feature}", fontsize=FONT_SIZE_TITLE)
        plt.ylabel(f"{feature}", fontsize=FONT_SIZE_AXES)
        plt.xlabel(f"Date", fontsize=FONT_SIZE_AXES)
        plt.tick_params(axis="both", labelsize=FONT_SIZE_TICKS)
        plt.show()

    turbine_selection = widgets.Dropdown(
        options=df.TurbID.unique(),
        value=df.TurbID.unique()[-1],
        description="Turbine ID",
    )

    feature_selection = widgets.Dropdown(
        options=features,
        description="Feature",
    )

    dates = pd.date_range(datetime(2020, 5, 1), datetime(2020, 12, 31), freq="D")

    options = [(date.strftime("%b %d"), date) for date in dates]
    index = (0, len(options) - 1)

    date_slider_selection = widgets.SelectionRangeSlider(
        options=options,
        index=index,
        description="Date (2020)",
        orientation="horizontal",
        layout={"width": "550px"},
    )

    fix_temps_button = widgets.Checkbox(
        value=False, description="Fix Temperatures", disabled=False
    )

    interact(
        plot_time_series,
        turbine=turbine_selection,
        feature=feature_selection,
        date_range=date_slider_selection,
        fix_temps=fix_temps_button,
    )


def time_series_turbine_pair(original_df: pd.core.frame.DataFrame, features: List[str]):
    """Creates interactive plots for the time series for a pair of turbines in the dataset.

    Args:
        original_df (pd.core.frame.DataFrame): The data used.
        features (List[str]): Features to include in the plot.
    """

    def plot_time_series(turbine_1, turbine_2, feature, date_range, fix_temps):
        df = original_df
        if fix_temps:
            df_2 = original_df.copy(deep=True)

            min_etemp = df_2["Etmp (°C)"].quantile(0.01)
            df_2["Etmp (°C)"] = df_2["Etmp (°C)"].apply(
                lambda x: np.nan if x < min_etemp else x
            )
            df_2["Etmp (°C)"] = df_2["Etmp (°C)"].interpolate()
            min_itemp = df_2["Itmp (°C)"].quantile(0.01)
            df_2["Itmp (°C)"] = df_2["Itmp (°C)"].apply(
                lambda x: np.nan if x < min_itemp else x
            )
            df_2["Itmp (°C)"] = df_2["Itmp (°C)"].interpolate()
            df = df_2

        data_1 = df[df.TurbID == turbine_1]
        data_1 = data_1[data_1.Datetime > date_range[0]]
        data_1 = data_1[data_1.Datetime < date_range[1]]

        data_2 = df[df.TurbID == turbine_2]
        data_2 = data_2[data_2.Datetime > date_range[0]]
        data_2 = data_2[data_2.Datetime < date_range[1]]

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 5))
        ax1.plot(data_1["Datetime"], data_1[feature], "-")
        ax1.set_title(f"Time series of {feature} for turbine {turbine_1}", fontsize=FONT_SIZE_TITLE)
        ax2.plot(data_2["Datetime"], data_2[feature], "-", c="green")
        ax2.set_title(f"Time series of {feature} for turbine {turbine_2}", fontsize=FONT_SIZE_TITLE)
        ax1.set_ylabel(f"{feature}", fontsize=FONT_SIZE_AXES)
        ax2.set_ylabel(f"{feature}", fontsize=FONT_SIZE_AXES)
        ax1.set_xlabel(f"Date", fontsize=FONT_SIZE_AXES)
        ax2.set_xlabel(f"Date", fontsize=FONT_SIZE_AXES)
        plt.tick_params(axis="both", labelsize=FONT_SIZE_TICKS)
        plt.tight_layout()
        plt.show()

    turbine_selection_1 = widgets.Dropdown(
        options=original_df.TurbID.unique(),
        value=original_df.TurbID.unique()[-2],
        description="Turbine ID",
    )

    turbine_selection_2 = widgets.Dropdown(
        options=original_df.TurbID.unique(),
        value=original_df.TurbID.unique()[-1],
        description="Another Turbine ID",
        style={"description_width": "initial"},
    )

    feature_selection = widgets.Dropdown(
        options=features,
        description="Feature",
    )

    fix_temps_button = widgets.Checkbox(
        value=False, description="Fix Temperatures", disabled=False
    )

    dates = pd.date_range(datetime(2020, 5, 1), datetime(2020, 12, 31), freq="D")

    options = [(date.strftime("%b %d"), date) for date in dates]
    index = (0, len(options) - 1)

    date_slider_selection = widgets.SelectionRangeSlider(
        options=options,
        index=index,
        description="Date (2020)",
        orientation="horizontal",
        layout={"width": "550px"},
    )

    interact(
        plot_time_series,
        turbine_1=turbine_selection_1,
        turbine_2=turbine_selection_2,
        feature=feature_selection,
        date_range=date_slider_selection,
        fix_temps=fix_temps_button,
    )

    
def plot_pairplot(
    original_df: pd.core.frame.DataFrame,
    turb_id: int,
    features: List[str],
    fraction: float=0.01
):
    """Creates a pairplot of the features.

    Args:
        df (pd.core.frame.DataFrame): The data used.
        turb_id (int): Selected turbine ID
        features (List[str]): List of features to include in the plot.
        fraction (float): amount of data to plot, to reduce time.
    """
    data_single_turbine = original_df[original_df.TurbID==turb_id][features]
    data_single_turbine = data_single_turbine.sample(frac=fraction)
    with sns.plotting_context(rc={"axes.labelsize":20}):
        sns.pairplot(data_single_turbine)
    plt.show()