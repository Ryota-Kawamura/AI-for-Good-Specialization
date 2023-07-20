import os, io
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import ipywidgets as widgets

from ipywidgets import interact
from sklearn.metrics import confusion_matrix
from PIL import Image as ImageOps, ImageEnhance
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, precision_score, recall_score
import plotly.graph_objects as go

import pickle
import pandas as pd
import plotly.express as px
from plotly.subplots import make_subplots
import random

from typing import List, Dict, Callable


def data_aug_flip(image: tf.Tensor):
    '''Interactive function to illustrate flipping of images.
    
    Args:
       image (tf.Tensor): An image to be displayed
    '''    
    img = ImageOps.fromarray(np.uint8(image*250), mode="RGB")

    def _data_aug_flip(random_flip='horizontal'):
        plt.figure(figsize=(10, 10))
        ax = plt.subplot(1, 2, 1)
        ax.imshow(img)
        ax.set_title('Original')
        plt.axis("off")

        if random_flip == 'horizontal':
            img2 = img.transpose(0)
        if random_flip == 'vertical':
            img2 = img.transpose(1)
        if random_flip == 'horizontal_and_vertical':
            img2 = img.transpose(0)
            img2 = img2.transpose(1)
        ax = plt.subplot(1, 2, 2)
        ax.imshow(img2)
        ax.set_title('Flipped')
        plt.axis("off")

    random_flip_widget = widgets.RadioButtons(
        options=['horizontal', 'vertical', 'horizontal_and_vertical'],
        value='horizontal',
        layout={'width': 'max-content'}, # If the items' names are long
        description='Random Flip',
        disabled=False,
        style = {'description_width': 'initial'},
    )
    
    interact(_data_aug_flip, random_flip = random_flip_widget)
    
    
def data_aug_zoom(image: tf.Tensor):
    '''Interactive function to illustrate zooming images.
    
    Args:
       image (tf.Tensor): An image to be displayed
    '''
    img = ImageOps.fromarray(np.uint8(image*250), mode="RGB")

    def _data_aug_zoom(zoom_factor):
        plt.figure(figsize=(10, 10))
        ax = plt.subplot(1, 2, 1)
        ax.imshow(img)
        ax.set_title('Original')
        plt.axis("off")
        w, h = img.size
        zoom2 = zoom_factor * 2
        x, y = int(w / 2), int(h / 2)
        img2 = img.crop(( x - w / zoom2, y - h / zoom2, 
                        x + w / zoom2, y + h / zoom2))

        img2 = img2.resize((w, h), ImageOps.LANCZOS)
        ax = plt.subplot(1, 2, 2)
        ax.imshow(img2)
        ax.set_title('Zoomed')
        plt.axis("off")

    zoom_widget = widgets.FloatSlider(
        value=1,
        min=1,
        max=2,
        step=0.05,
        description='Zoom: ',
        disabled=False,
        continuous_update=False,
        orientation='horizontal',
        readout=True,
        readout_format='.1f',
        style = {'description_width': 'initial'}
    )
    interact(_data_aug_zoom, zoom_factor = zoom_widget)


def data_aug_rot(image: tf.Tensor):
    '''Interactive function to illustrate rotating images.
    
    Args:
       image (tf.Tensor): An image to be displayed
    '''
    img = ImageOps.fromarray(np.uint8(image*250), mode="RGB")
    
    def _data_aug_rot(angle):
        plt.figure(figsize=(10, 10))
        ax = plt.subplot(1, 2, 1)
        ax.imshow(img)
        ax.set_title('Original')
        plt.axis("off")
        
        img2 = img.rotate(angle)
        ax = plt.subplot(1, 2, 2)
        ax.imshow(img2)
        ax.set_title('Rotated')
        plt.axis("off")

    angle_widget = widgets.FloatSlider(
        value=0,
        min=-40,
        max=40,
        step=5,
        description='Rotation (deg): ',
        disabled=False,
        continuous_update=False,
        orientation='horizontal',
        readout=True,
        readout_format='.1f',
        style = {'description_width': 'initial'},

    )
    interact(_data_aug_rot, angle = angle_widget)


def data_aug_brightness(image: tf.Tensor):
    '''Interactive function to illustrate contrasting images.
    
    Args:
       image (tf.Tensor): An image to be displayed
    '''
    img = ImageOps.fromarray(np.uint8(image*250), mode="RGB")
    enhancer = ImageEnhance.Brightness(img)

    def _data_aug_brightness(brightness_factor):
        plt.figure(figsize=(10, 10))
        ax = plt.subplot(1, 2, 1)
        ax.imshow(img)
        ax.set_title('Original')
        plt.axis("off")
        
        img2 = enhancer.enhance(brightness_factor)
        ax = plt.subplot(1, 2, 2)
        ax.imshow(img2)
        ax.set_title('Brightness')
        plt.axis("off")

    brightness_widget = widgets.FloatSlider(
        value=1,
        min=0.5,
        max=1.5,
        step=0.2,
        description='Brightness factor: ',
        disabled=False,
        continuous_update=False,
        orientation='horizontal',
        readout=True,
        readout_format='.1f',
        style = {'description_width': 'initial'},
    )
    interact(_data_aug_brightness, brightness_factor = brightness_widget)


def data_aug_random(image: tf.Tensor):
    '''Interactive function to illustrate combined data augmentation steps.
    
    Args:
       image (tf.Tensor): An image to be displayed
    '''
    img = ImageOps.fromarray(np.uint8(image*250), mode="RGB")

    def _data_aug_randomimage(t1, t2, t3, t4):
        plt.figure(figsize=(12, 3))
        ax = plt.subplot(1, 4, 1)
        ax.imshow(img)
        ax.set_title('Original')
        plt.axis("off")
        
        for i in range(3):
            transposition = random.choice([0, 1])
            brightness_factor = random.uniform(0.6, 1.4)
            zoom_factor = random.uniform(1, 2)
            angle = random.uniform(-45, 45)
            
            img2 = img
            if t1:
                img2 = img2.transpose(transposition)
            if t3:
                enhancer = ImageEnhance.Brightness(img2)
                img2 = enhancer.enhance(brightness_factor)
            if t2:
                img2 = img2.rotate(angle)
            if t4:
                w, h = img2.size
                zoom2 = 2 * zoom_factor
                x, y = int(w / 2), int(h / 2)
                img2 = img2.crop(( x - w / zoom2, y - h / zoom2, 
                                x + w / zoom2, y + h / zoom2))
                img2 = img2.resize((w, h), ImageOps.LANCZOS)
                
            ax = plt.subplot(1, 4, i + 2)
            ax.imshow(img2)
            ax.set_title('Augmented')
            plt.axis("off")

    t1 = widgets.Checkbox(
        value=True,
        description='RandomFlip',
        disabled=False,
        indent=False
    )
    t2 = widgets.Checkbox(
        value=False,
        description='RandomRotation',
        disabled=False,
        indent=False
    )
    t3 = widgets.Checkbox(
        value=False,
        description='RandomContrast',
        disabled=False,
        indent=False
    )
    t4 = widgets.Checkbox(
        value=False,
        description='RandomZoom',
        disabled=False,
        indent=False
    )
    
    interact(_data_aug_randomimage, t1 = t1, t2 = t2, t3 = t3, t4 = t4)

    return _data_aug_randomimage


def get_performance_metrics(y_true: np.ndarray, y_scores: np.ndarray) -> List:
    '''Calculates the Accuracy, Precision and Recall for the given pair of labels and prediction scores
    
    Args:
        y_true (np.ndarray): an array of true y values
        y_scores (np.ndarray): an array of predicted y values
        
    Returns:
        _ (List[float]): [accuracy, precision, recall]
    '''
    y_pred_1 = (y_scores >= 0.5) * 1
    return [accuracy_score(y_true, y_pred_1),
            precision_score(y_true, y_pred_1),
            recall_score(y_true, y_pred_1)]


def display_predictions(model: tf.keras.Model, label2cat: Dict[int, str], files_list: List[str]) -> Callable:
    '''Displays the predictions with a gauge and image
    
    Args:
        model (tf.keras.Model):
        label2cat (dict): Mapping from labels to categories
        y_scores (np.array): an array of predicted y values
        files_list (List[str]): list of file names
        
    Returns:
        _ (Callable): A function that plots the predictions
    '''    
        
    def plot_img_and_gradcam(img_array, axes, fig, label, mutator):   
        img_array[0] = mutator(img_array[0])
        prediction = model.predict(img_array / 255., verbose = 0)[0][0]
        predicted_label = label2cat[(prediction > 0.5) * 1]

        # In the first box, plot the gauge
        image_data = create_gauge_chart(prediction, predicted_label)
        axes[0].imshow(image_data)
        axes[0].axis('off')
        #axes[0].set_title(f"Predicted: {predicted_label}")
        # In the second box, plot the original image
        axes[1].imshow(img_array[0].astype("uint8"))
        axes[1].set_title(f"True: {label}")
        axes[1].axis('off')
        
        
    def create_gauge_chart(current_value, predicted_label):
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
                width=150,
                height=150,
                paper_bgcolor=plot_bgcolor,
                annotations=[
                    go.layout.Annotation(
                        text=f"<b>Model output:</b><br>{((int(current_value*100)/100))}<br><b>Prediction:</b><br>{predicted_label}",
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

        return ImageOps.open(buf)

    
    def plot_example_by_index(file_index):

        fig, axes = plt.subplots(1, 2, figsize=(8, 3.4))
        image_path = files_list[file_index]
        image = get_img_array(image_path, size=(150, 150))
        image_class = image_path.split('/')[-2]
        plot_img_and_gradcam(image, axes, fig, image_class, lambda x: x)

        plt.show()
        
    return plot_example_by_index


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

    
def get_img_array(img_path: str, size: int) -> np.ndarray:
    '''Gets an image array from the specified path
    
    Args:
        img_path (str): path tho th file
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


def plot_training_history(filename: str):
    ''' Plots the training history stored in a pickle file
    
    Args:
        filename (str): the path to the file
    '''
    # open a file, where you stored the pickled data
    with open(filename, 'rb') as file:
        history_training = pd.DataFrame(data=pickle.load(file))

    fig = make_subplots(rows=1, cols=2)
    epochs = list(range(history_training.shape[0]))

    fig.add_trace(go.Scatter(x=epochs, y=history_training['accuracy'],
                        mode='lines',
                        name='Training accuracy'), row=1, col=1)
    fig.add_trace(go.Scatter(x=epochs, y=history_training['val_accuracy'],
                        mode='lines',
                        name='Validation accuracy'), row=1, col=1)
    fig.add_trace(go.Scatter(x=epochs, y=history_training['loss'],
                        mode='lines',
                        name='Training loss'), row=1, col=2)
    fig.add_trace(go.Scatter(x=epochs, y=history_training['val_loss'],
                        mode='lines',
                        name='Validation loss'), row=1, col=2)

    # Update xaxis properties
    fig.update_xaxes(title_text="epochs", row=1, col=1)
    fig.update_xaxes(title_text="epochs", row=1, col=2)

    # Update yaxis properties
    fig.update_yaxes(title_text="Accuracy", row=1, col=1)
    fig.update_yaxes(title_text="Loss", row=1, col=2)

    # Update title and height
    fig.update_layout(title_text="Training metrics", height=400)
    fig.show()
    
    
def get_some_images(where: str, n: int=20):
    '''Gets some image paths.
    
    Args:
        where (str): Image location
        n (int): number of images of each class to return
    '''
    dataset_dir = './data/' + where + '/'
    no_damage = list(set(os.listdir(dataset_dir + 'no_damage')))[0:n]
    image_paths = [dataset_dir + 'no_damage/' + i for i in no_damage]
    damage = list(set(os.listdir(dataset_dir + 'visible_damage')))[0:n]
    image_paths.extend([dataset_dir + 'visible_damage/' + i for i in damage])
    return image_paths, 2 * n


def display_confusion_matrix(y_labels: np.ndarray, y_predict_prob_1: np.ndarray):
    '''Display a confusion matrix
    
    Input:
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
    
    
def find_misclassified_images(
    y_labels: np.ndarray,
    y_predict_prob_1: np.ndarray,
    filenames: List[str],
    original_dataset_dir: str
) -> List[str]:
    '''Finds images that were misclassified.
    
    Input:
        y_labels (np.array): An array like with the true lables
        y_predict_prob_1 (np.array): An array like with the predictions
        filenames (List[str]): A list of filenames
        original_dataset_dir (str): path to file directory
    
    Returns:
        misclassified (List[str]): list of paths to misclassified images
    '''
    y_predict_prob_1 = ((np.array(y_predict_prob_1) > 0.5) * 1).reshape(-1,)
    # Get the examples where the model fails
    ids = np.where([a != b for a, b in zip(y_labels, y_predict_prob_1)])

    misclassified = [original_dataset_dir + filenames[idx] for idx in ids[0]]
    
    return misclassified


def plot_one_image(image: np.ndarray, label: str):
    '''Plots the image with the given title
    
    Args:
        image (np.ndarray): Array like with image information
        label (str): the title to be displayed with the image
    '''
    plt.imshow(image)
    plt.title(label)
    plt.axis("off")
    
    plt.show()