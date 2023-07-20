import shutil
import numpy as np 
import math
import os, io
import pandas as pd
from termcolor import colored

import tensorflow as tf
from tensorflow.keras.models import Model
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, precision_score, recall_score

import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from PIL import Image
import plotly.graph_objects as go

import ipywidgets as widgets
from ipywidgets import interact
from IPython.display import clear_output

import folium 
from folium.plugins import FastMarkerCluster
from typing import List, Dict, Tuple, Callable


def show_predictions_on_map(
    predictions: np.ndarray,
    filenames: List[str],
    probability_buckets: Dict[str, Tuple[float, float]]
) -> folium.Map:
    '''Create a plot to visualize two sets of geo points: the ones with satellite
    images of damage and ones with satellite images of no damage done by hurricane.
    
    Args:
        predictions (np.ndarray): array of predictions for images
        filenames (List[str]): list of image filenames
        probability_buckets (dict): dictionary of colors mapping to the probability buckets
    Returns:
        map3 (folium.Map): The map with the points
    '''
    predictions = predictions.reshape(-1)
    
    def is_contained(value, interval):
         return value >= interval[0] and value < interval[1]
    
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
    
    
    icon_create_function0= icon_creator("small")   
    icon_create_function1 = icon_creator("medium")
    icon_create_function2 = icon_creator("large")

    points = [filename.split('/')[1].replace('.jpeg', '').split("_") for filename in filenames]
    points = [[float(point[1]), float(point[0])] for point in points]
    map3 = folium.Map(location=[points[0][0], points[0][1]], tiles='CartoDB positron', zoom_start=6)

    marker_cluster0 = FastMarkerCluster([], icon_create_function=icon_create_function0).add_to(map3)
    marker_cluster1 = FastMarkerCluster([], icon_create_function=icon_create_function1).add_to(map3)
    marker_cluster2 = FastMarkerCluster([], icon_create_function=icon_create_function2).add_to(map3)

    for i, point in enumerate(points):
        filename = filenames[i]
        prediction = predictions[i]
        file_location = f"data/test/{filename}"
        if prediction > 0.5:
            predicted_class = 'visible_damage'
        else:
            predicted_class = 'no_damage'
        gauge = create_gauge_chart(prediction, predicted_class, 128)
        im = Image.open(file_location)
        # Create a new image with gauge and sattelite image and save it
        new_im = Image.new('RGB', (256, 128))
        new_im.paste(gauge, (0, 0))
        new_im.paste(im, (128, 0))
        file_location_gauge = f"images/gauge/{filename.replace('jpeg', 'png')}"
        new_im.save(file_location_gauge)
        # Create the html with the location of the saved image for the popup
        popup_text = f"<img src='{file_location_gauge}'>"
        
        if is_contained(prediction, probability_buckets['green']):
            folium.Marker(point, popup=popup_text, icon=folium.Icon(color="green")).add_to(marker_cluster0)
        if is_contained(prediction, probability_buckets['orange']):
            folium.Marker(point, popup=popup_text, icon=folium.Icon(color='orange')).add_to(marker_cluster1)
        if is_contained(prediction, probability_buckets['red']):
            folium.Marker(point, popup=popup_text, icon=folium.Icon(color="red")).add_to(marker_cluster2)
            
    return map3
    

def display_confusion_matrix(y_labels: np.ndarray, y_predict_prob_1: np.ndarray):
    '''Display a confusion matrix
    
    Args:
        y_labels (np.ndarray): An array like with the true lables
        y_predict_prob_1 (np.ndarray): An array like with the predictions
    '''
    confusion_matrix_1 = tf.math.confusion_matrix(
        y_labels.reshape(-1),
        (y_predict_prob_1.reshape(-1) >= 0.5)*1, # Convert probabilities to 0 and 1 labels
        num_classes=2
    ).numpy()

    disp = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix_1,
                                  display_labels=[ 'No damage', 'Visible damage'])

    disp.plot(cmap="Blues", values_format='')


# Rebuild the model to get the outputs that we need
def gradcam_model_extractor(
    model_r: tf.keras.Model,
    last_conv_layer_id: int
) -> tf.keras.Model:
    '''Gets the gradcam model
    
    Args:
        model_r (tf.keras.Model): Original model
        last_conv_layer_id (int): Layer in the original model on which to calculate gradcam
    '''
    
    inputs = model_r.input
    return Model(inputs=inputs, outputs=[model_r.layers[last_conv_layer_id].output, model_r.output])

def get_img_array(img_path: str, size: int) -> np.ndarray:
    '''Gets an image array from the specified path
    
    Args:
        img_path (str): path tho the file
        size (int): image size
    
    Returns:
        array (np.ndarray): array with image
    '''
    
    img = tf.keras.preprocessing.image.load_img(img_path, target_size=size)
    # `array` is a float32 Numpy array of shape (size, size, 3)
    array = tf.keras.preprocessing.image.img_to_array(img)
    # We add a dimension to transform our array into a "batch"
    # of size (1, size, size, 3)
    array = np.expand_dims(array, axis=0)
    
    return array

def make_gradcam_heatmap(
    img_array: np.ndarray,
    model: tf.keras.Model, 
    last_conv_layer_name: int,
) -> np.ndarray:
    '''Creates the gradcam heatmap
    
    Args:
        array (np.ndarray): array with image
        model (tf.keras.Model): Original model
        last_conv_layer_id (int): Layer in the original model on which to calculate gradcam
        
    Returns:
        _ (np.ndarray): an array containing the heatmap
    '''
    
    # First, we create a model that maps the input image to the activations
    # of the last conv layer as well as the output predictions
    grad_model = gradcam_model_extractor(model, last_conv_layer_name)
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array / 255.)
        class_channel = preds
        if preds[0] < 0.5:
            class_channel = -preds# - preds

    # This is the gradient of the output neuron (top predicted or chosen)
    # with regard to the output feature map of the last conv layer
    grads = tape.gradient(class_channel, last_conv_layer_output)
    #print(grads)

    # This is a vector where each entry is the mean intensity of the gradient
    # over a specific feature map channel
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    #print('pooled')
    #print(pooled_grads[..., tf.newaxis])

    # We multiply each channel in the feature map array
    # by "how important this channel is" with regard to the top predicted class
    # then sum all the channels to obtain the heatmap class activation
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # For visualization purpose, we will also normalize the heatmap between 0 & 1
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()


# Create super-imposed gradcam
def display_gradcam(
    img: np.ndarray,
    heatmap: np.ndarray,
    color_map: str,
    alpha: float=0.4
) -> np.ndarray:
    '''Superimposes the heatmap over the original image
    
    Args:
        img (np.ndarray): array with image
        heatmap (np.ndarray): an array containing the heatmap
        color_map (str): The color map to be used for heatmap
        alpha (float): heatmap transparency
        
    Returns:
        superimposed_img (np.ndarray): an array containing the image with heatmap
    '''
    # Rescale heatmap to a range 0-255
    heatmap = np.uint8(255 * heatmap)

    # Use jet colormap to colorize heatmap
    jet = cm.get_cmap(color_map)

    # Use RGB values of the colormap
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]

    # Create an image with RGB colorized heatmap
    jet_heatmap = tf.keras.preprocessing.image.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
    jet_heatmap = tf.keras.preprocessing.image.img_to_array(jet_heatmap)

    # Superimpose the heatmap on original image
    superimposed_img = jet_heatmap * alpha + img
    superimposed_img = tf.keras.preprocessing.image.array_to_img(superimposed_img)
    return superimposed_img


def display_predictions_gradcam(
    model: tf.keras.Model,
    label2cat: Dict[int, str],
    files_list: List[str],
    base_path: str,
    class_name1: str='',
    class_name2: str=''
):
    '''Displays the predictions with a gauge, image and image with gradcam
    
    Args:
        model (tf.keras.Model):
        label2cat (dict): Mapping from labels to categories
        files_list (List[str]): list of file names
        base_path (str): path to the folder with files
        class_name1 (str): Name of the first class
        class_name2 (str): Name of the second class
        
    Returns:
        plot_example_by_index (Callable): A function that plots the predictions
    '''  
    class_name1, class_name2 = label2cat[0], label2cat[1]
    last_conv_layer_name = 8
    
    dir_class1 = os.path.join(base_path, class_name1)
    dir_class2 = os.path.join(base_path, class_name2)
    
    
    def plot_original_image(img_array, ax, label):
        ax.imshow(img_array[0].astype("uint8"))
        ax.set_title(f"True: {label}")
        ax.axis('off')


    def plot_grad_cam_image(img_array, ax, plt, label):
        heatmap = make_gradcam_heatmap(img_array, model, last_conv_layer_name)
        color_map = 'cool'
        superimposed_img = display_gradcam(img_array[0], heatmap, color_map)
        im = ax.imshow(superimposed_img, cmap=color_map)

        ax.set_title(f'Prediction: {label}')
        ax.axis('off')
        
        # create an axes on the right side of ax. The width of cax will be 5%
        # of ax and the padding between cax and ax will be fixed at 0.05 inch.
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cbar = plt.colorbar(im, cax=cax)
        cbar.set_ticks([im.colorbar.vmin, im.colorbar.vmax])
        cbar.set_ticklabels(["low", "high"])

        
    def plot_gauge(probability, predicted_label, ax):
        image_data = create_gauge_chart(probability, predicted_label)
        ax.imshow(image_data)
        ax.axis('off')

        
    def plot_img_and_gradcam(img_array, axes, fig, folder, mutator):
        img_array[0] = mutator(img_array[0])

        probability = model.predict(img_array / 255., verbose = 0)[0][0]
        predicted_label = label2cat[(probability > 0.5) * 1]
        prediction = {
          "category": predicted_label,
          "probability": probability
        }

        # In the first box, plot the gauge
        plot_gauge(probability, predicted_label, axes[0])
        # In the second box, plot the original image
        plot_original_image(img_array, axes[1], folder)
        # In the third box, plot the superimposed image
        plot_grad_cam_image(img_array, axes[2], fig, predicted_label)
        
    
    def plot_example_by_index(file_index):
        matches = files_list
        fig, axes = plt.subplots(2, 3, figsize=(12, 3.4 * 2))
       
        im_class1 = get_img_array(os.path.join(dir_class1, matches[file_index]), size=(150, 150))
        plot_img_and_gradcam(im_class1, axes[0], fig, class_name1, lambda x: x)

        im_class2 = get_img_array(os.path.join(dir_class2, matches[file_index]), size=(150, 150))
        plot_img_and_gradcam(im_class2, axes[1], fig, class_name2, lambda x: x)

        plt.show()
        
    return plot_example_by_index


def find_matching_images(path) -> List[str]:
    '''Finds images that are recorded with and without damage at the same location.
    
    Input:
        path (str): path to the images folder
    
    Returns:
        matches (List[str]): list of paths to matching images
    '''
    test_damage_dir = path + 'visible_damage'
    test_nodamage_dir = path + 'no_damage'

    matches = list(set(os.listdir(test_nodamage_dir)).intersection(os.listdir(test_damage_dir)))
    print(f"There are {len(matches)} images in the set")
    
    return matches


def interact_with_slider(func: Callable, slider_min: int, slider_max: int, *args, **kwargs):
    '''Interactive function for creating a dropdown menu to select filters for the dataset.

    Args:
        function (Callable): A function to be wrapped.
        slider_min (int): The minimum of the slider range
        slider_max (int): The maximum of the slider range
        *args: additional parameters for the function
        **kwargs: additional keyword parameters for the function
    '''
    # Create a slider
    file_index_widget = widgets.IntSlider(min=slider_min, max=slider_max, description='Image index')  
    # Create interactive output
    interact(
        func(*args, **kwargs),
        file_index=file_index_widget
    )
    

def get_performance_metrics(y_true: np.ndarray, y_scores: np.ndarray) -> Tuple[float]:
    '''Calculates the Accuracy, Precision and Recall for the given pair of labels and prediction scores
    
    Args:
        y_true (np.ndarray): an array of true y values
        y_scores (np.ndarray): an array of predicted y values
        
    Returns:
        _ (Tuple[float]): [accuracy, precision, recall]
    '''
    y_pred_1 = (y_scores >= 0.5) * 1
    return (accuracy_score(y_true, y_pred_1),
            precision_score(y_true, y_pred_1),
            recall_score(y_true, y_pred_1))


def create_gauge_chart(current_value: float, predicted_class: str, img_size: int=150) -> Image:
    '''Creates a gauge chart showing the prediction on a scale from 0 to 1.
    
    Args:
        current_value (float): The predicted value
        predicted_class (str): The predicted class
        img_size (int): The size of image
    
    Returns:
        _ (Image): Image with a gauge
    '''
    plot_bgcolor = "#def"

    quadrant_colors = [plot_bgcolor, "#f25829", "#f2a529", "#eff229", "#85e043", "#2bad4e"] 
    quadrant_text = ["", "<b>1.0</b>", "<b></b>", "<b>0.5</b>", "<b></b>", "<b>0.0</b>"]
    n_quadrants = len(quadrant_colors) - 1

    min_value = 0
    max_value = 1
    hand_length = np.sqrt(2) / 4
    hand_angle = np.pi * (1 - (max(min_value, min(max_value, current_value)) - min_value) / (max_value - min_value))

    fig = go.Figure(
        data=[
            go.Pie(
                values=[0.5] + (np.ones(n_quadrants) / 2 / n_quadrants).tolist(),
                rotation=90,
                hole=0.5,
                marker_colors=quadrant_colors,
                text=quadrant_text,
                textinfo="text",
                hoverinfo="skip",
            ),
        ],
        layout=go.Layout(
            showlegend=False,
            margin=dict(b=0,t=10,l=10,r=10),
            width=img_size,
            height=img_size,
            paper_bgcolor=plot_bgcolor,
            annotations=[
                go.layout.Annotation(
                    text=f"<b>Model output:</b><br>{((int(current_value*100)/100))}<br><b>Prediction:</b><br>{predicted_class}",
                    x=0.5, xanchor="center", xref="paper",
                    y=0.25, yanchor="middle", yref="paper",
                    showarrow=False,
                )
            ],
            shapes=[
                go.layout.Shape(
                    type="circle",
                    x0=0.48, x1=0.52,
                    y0=0.48, y1=0.52,
                    fillcolor="#333",
                    line_color="#333",
                ),
                go.layout.Shape(
                    type="line",
                    x0=0.5, x1=0.5 + hand_length * np.cos(hand_angle),
                    y0=0.5, y1=0.5 + hand_length * np.sin(hand_angle),
                    line=dict(color="#333", width=4)
                )
            ]
        )
    )

    buf = io.BytesIO(fig.to_image(format="png"))

    return Image.open(buf)
