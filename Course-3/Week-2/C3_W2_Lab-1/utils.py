import ipywidgets as widgets

from ipywidgets import HBox
import folium 
from folium.plugins import FastMarkerCluster
import matplotlib.pyplot as plt
from PIL import Image
import os
import numpy as np 
import pandas as pd

from IPython.display import clear_output, display
from ipyfilechooser import FileChooser

from typing import List, Dict, Tuple, Any, Callable


def images_on_server(display: display) -> Dict['str', Any]:
    ''' Display a file chooser widget that allows to select files that are in the server side

    Args:
        display (display): The widget display to render the ouputs
    Returns:
        _ (Dict): Dictionary that returns the filechooser and main display
    '''
    fc = FileChooser('./data/test/')
    
    main_display1 = widgets.Output()
    main_display2 = widgets.Output()
    main_display = HBox([main_display1, main_display2])
    
    def show_example(chooser):
        filename = chooser.selected
        tokens = filename.split("/")
        lon, lat = tokens[-1].replace(".jpeg", "").split("_")
        with  main_display1:
            main_display1.clear_output()
            print(f"set: { tokens[-3]}")
            print(f"class: { tokens[-2]}")
            print(f"lat: {lat}")
            print(f"lon: {lon}")
            image = Image.open(chooser.selected)
            display(image) 
                         
        with  main_display2:
            main_display2.clear_output()
            mapit = folium.Map(width=300,height=300,location=[lat, lon], zoom_start=7 )
            folium.Marker( location=[lat, lon], fill_color='#43d9de', radius=8 ).add_to( mapit )
            display(mapit)

    fc.register_callback(show_example)
    
    return {'fileChooser': fc, 'output': main_display}


def leaflet_plot(n_samples: int=2000) -> folium.Map:
    '''Create a plot to visualize two sets of geo points: the ones with satellite
    images of damage and ones with satellite images of no damage done by hurricane.
    
    Args:
        n_samples (int): Number of points to show on the plot for each class
    Returns:
        map3 (folium.Map): The map with the points
    '''
    
    def icon_creator(size):
        return """
        function(cluster) {
          var childCount = cluster.getChildCount(); 
          var c = ' marker-cluster-';
          return new L.DivIcon({ html: '<div><span>' + childCount + '</span></div>', 
                                 className: 'marker-cluster'+c, 
                                 iconSize: new L.Point(40, 40) });
        }
        """.replace('marker-cluster-', f'marker-cluster-{size}')
    
    icon_create_function1 = icon_creator("large")
    icon_create_function2 = icon_creator("small")
    
    # Load datasets with n_samples of each label
    train_damage = load_coordinates('./data/train/visible_damage', n_samples)
    train_nodamage = load_coordinates('./data/train/no_damage', n_samples)
    
    map3 = folium.Map(location=[train_damage[0][1][0], train_damage[0][1][1]], tiles='CartoDB positron', zoom_start=6)

    marker_cluster = FastMarkerCluster([], icon_create_function=icon_create_function2).add_to(map3)
    for filename, point in train_nodamage:
        popup_text = f"No damage\n<img src='data/train/no_damage/{filename}'>"
        folium.Marker(point, popup=popup_text, icon=folium.Icon(color="green")).add_to(marker_cluster)

    marker_cluster2 = FastMarkerCluster([], icon_create_function=icon_create_function1).add_to(map3)
    for filename, point in train_damage:
        popup_text = f"Visible damage\n<img src='data/train/visible_damage/{filename}'>"
        folium.Marker(point, popup=popup_text, icon=folium.Icon(color="red")).add_to(marker_cluster2)

    return map3


def interactive_plot_pair(base: str, matches: List[str]) -> Callable:
    '''Create a plot to visualize a pair of images at the same location. One
    showing damage and the other showing no damage.
    
    Args:
        base (str): The base of the image path
        matches (List[str]): a list of image names
    Returns:
        plot_image_pairs (Callable): A function that plots the image pairs given the image index
    '''
    def plot_pairs(base, matches, index):
        fig = plt.figure(figsize=(12, 12))
        ax = []

        im_damage = Image.open(os.path.join(base, 'visible_damage', matches[index])).resize((200, 200))
        im_nodamage = Image.open(os.path.join(base, 'no_damage', matches[index])).resize((200, 200))

        ax.append(fig.add_subplot(1, 2, 1))
        ax[-1].set_title("No damage") 
        ax[-1].axis('off')
        plt.imshow(im_nodamage)

        ax.append(fig.add_subplot(1, 2, 2))
        ax[-1].set_title("Visible damage") 
        ax[-1].axis('off')
        plt.imshow(im_damage)
        plt.axis('off')
        plt.show()


    def plot_image_pairs(file_index):
        plot_pairs(base, matches, index=file_index)
        
    return plot_image_pairs
    

def load_coordinates(path: str, samples: int) -> List[Tuple[str, Tuple[float, float]]]:
    '''Load the  GPS coordinates from the first few samples in a given folder
    
    Args:
        path (str): path to the images
        samples (int): number of samples to take
    
    Returns:
        coordinates: An array containing the GPS coordianates extracted from the filenames
    '''
    files = os.listdir(path)
    coordinates = []
    indexes = list(range(len(files)))
    np.random.shuffle(indexes)
    indexes = indexes[0:samples]
    for i in indexes:
        # Get the coordinates
        coordinate = files[i].replace('.jpeg', '').split('_')
        coordinates.append((files[i], (float(coordinate[1]) , float(coordinate[0]))))
        
    return coordinates


def get_dataframe_from_file_structure() -> pd.core.frame.DataFrame:
    ''' Creates a dataframe with metadata based on the file structure.
    
    Returns:
        _ (pd.core.frame.DataFrame): Dataframe with metadata
    '''
    # Dataset paths
    base = './data'
    subsets = ['train', 'validation', 'test']
    labels = ['visible_damage', 'no_damage']

    # Navigate through every folder and its contents to create a dataframe
    data = []
    for seti in subsets:
        for label in labels:
            files = os.listdir(os.path.join(base, seti, label))
            for filename in files:
                path = os.path.join(seti, label, filename)
                lon, lat = filename.replace(".jpeg", "").split("_")
                data.append([seti, label, lat, lon, path, filename])

    # Create dataframe
    return pd.DataFrame(data = data, columns=['subset', 'label', 'lat', 'lon', 'path', 'filename'])