import os
import pandas as pd 
from glob import glob
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import plotly.express as px
from typing import List
from matplotlib import cm, colors


def get_metadata(data_folder: str) -> pd.core.frame.DataFrame:
    '''Inspect the data_folder and create a data frame using the metainformation of the images.
    The folder name are the classes and place code must be extracted out of the file name.
    
    Args:
        data_folder (str): The location of the data.

    Returns:
        pd.core.frame.DataFrame: The dataframe with metadata.
    '''
    # Find all file names within the data folder.
    all_paths = [y for x in os.walk(data_folder) for y in glob(os.path.join(x[0], '*.JPG'))]

    meta_data_list = []
    for file_path in all_paths:
        # Split the path into subfolders and file name.
        data_folder_name, class_folder_name, file_name = file_path.split("/")
        # Camera locaion is given within the file name.
        camera_location = file_name.split('_')[2]
        meta_data_list.append([camera_location, class_folder_name, file_path])

    # Create a dataframe with metadata.
    meta_data = pd.DataFrame(data=meta_data_list, columns=['location', 'class', 'path'])
    
    return meta_data


def plot_donut_chart(class_counts: pd.core.frame.DataFrame):
    '''Plot a donut chart of the class distribution in the dataset.
    
    Args:
        class_counts (pd.core.frame.DataFrame): The dataframe with info about classes.
    '''
        
    fig = px.pie(
        pd.DataFrame({'class': class_counts.index, 'values': class_counts.values}),
        values='values',
        names='class',
        title='Distribution of Animals', 
        hole = 0.4
    )
    fig.update_traces(textposition='inside', textinfo='percent+label')
    fig.show()


def plot_bar_chart(meta_data: pd.core.frame.DataFrame):
    '''Plot a bar chart of the class distribution in the dataset.
    
    Args:
        meta_data (pd.core.frame.DataFrame): The dataframe with data about all data.
    '''
    cmap = colors.ListedColormap(cm.tab20c.colors + cm.tab20b.colors, name='tab40')
    
    cross = pd.crosstab(index=meta_data['location'], 
                        columns=meta_data["class"],
                        normalize='index')
    cross.plot(
        kind="bar", 
        stacked=True,
        figsize=(11, 6),
        fontsize=12,
        cmap=cmap
    )
    plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
    plt.title("Relative class distribution by location", fontsize=20)
    plt.xlabel("Location", fontsize=16)
    plt.ylabel("Relative class distribution", fontsize=16)

    plt.show()
    


def plot_random_images(meta_data: pd.core.frame.DataFrame, location: str):
    '''Plots a 3x3 grid of random images.
    
    Args:
        meta_data (pd.core.frame.DataFrame): The dataframe with data about all data.
        location (str): The location of the camera trap from which the images are taken.
    '''
    plt.figure(figsize=(15, 15))
    for sample_number in range(9):
        sample_row = meta_data[meta_data['location']==location].sample()
        path = sample_row.path.values[0]
        ax = plt.subplot(3, 3, sample_number + 1)
        plt.imshow(mpimg.imread(path))
        plt.title(sample_row['class'].values[0], fontsize=14)
        plt.axis("off")

        
def plot_images_from_all_locations(meta_data: pd.core.frame.DataFrame):
    '''Plots a 4x4 grid of images. Each image is a random image from a different location.
    
    Args:
        meta_data (pd.core.frame.DataFrame): The dataframe with data about all data.
    '''
    locations = sorted(meta_data['location'].unique())
    plt.figure(figsize=(15, 15))
    for loc_index, location in enumerate(locations):
        example = meta_data[meta_data['location']==location].sample()
        path = example.path.values[0]
        image = mpimg.imread(path)
        ax = plt.subplot(4, 4, loc_index + 1)
        plt.imshow(image)
        plt.title(f'loc_{location}_{example["class"].values[0]}', fontsize=14)
        plt.axis("off")
        

def plot_examples(sequence: List[str]):
    '''Plots a grid of specified images.
    
    Args:
        sequence (List[str]): A list of paths to images to be plotted.
    '''
    plt.figure(figsize=(15, 15))
    columns = 3
    rows = len(sequence) // columns + (len(sequence) % columns > 0)
    for index, example in enumerate(sequence):
        location=example.split("/")[2].split("_")[2]
        animal = example.split("/")[1]
        ax = plt.subplot(rows, columns, index + 1)
        plt.imshow(mpimg.imread(example))
        plt.title(f'loc_{location}_{animal}', fontsize=14)
        plt.axis("off")
