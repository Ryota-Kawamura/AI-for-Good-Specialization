import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob
import random
from random import choice
import logging
import shutil

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pickle

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.applications import nasnet
from tensorflow.keras.layers import Dense, Dropout, GlobalMaxPooling2D, Input
from tensorflow.keras.metrics import sparse_top_k_categorical_accuracy 

import visualization.visualization_utils as viz_utils

import seaborn as sns
import ipywidgets as widgets
from ipywidgets import interact, fixed

from sklearn.metrics import confusion_matrix

from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from PIL import Image as ImageOps, ImageEnhance, ImageDraw

from typing import Callable, List, Tuple, Dict, Optional, Any, Iterable
from detection.pytorch_detector import PTDetector


# Functions used in the first design phase notebook

def draw_bounding_box(image: ImageOps, megadetector_result: Dict):
    '''Draw a bounding on the given image, containing the first detected animal

    Args:
       image (ImageOps): A PIL image
       megadetector_result (dict): The MegaDetector output
       
    Returns:
       image2 (ImageOps): A PIL image
    '''
    image2 = image.copy()
    count_animals = 0
    # Loop over all detected objects in the image
    for detection in megadetector_result['detections']:
        # Only draw the box if the detected object is an animal
        if detection['category'] == '1':
            count_animals += 1
            # Get the bounding box
            bbox = detection['bbox']
            # Get the original image width and height
            w, h = image.size
            # Transform the bbox into a PIL rectangle [(x1, y1), (x2, y2)]
            # Pay attention to the pixels transformation
            shape = [(int(bbox[0]* w), int(bbox[1]*h)), (int((bbox[0]+bbox[2]) * w), int((bbox[1]+bbox[3])*h))]
            # Get the object to draw...
            img1 = ImageDraw.Draw(image2)  
            # Draw the rectangle into the copy of the original image
            img1.rectangle(shape, outline ="magenta")
    
    if count_animals == 0:
        print('Animals were not detected in this picture');

    # Return the modified copy of the image
    return image2


def crop_image(
    image: ImageOps,
    megadetector_result: Dict,
    input_folder: str,
    output_subfolder: str
) -> List:
    '''Crop the given image using the result from MegaDetector. Save all the crops into the given
    subfolder

    Args:
       image (ImageOps): A PIL image
       megadetector_result (dict): The MegaDetector output
       input_subfolder (str): A folder name to find the images
       output_subfolder (str): A folder name to save the cropped images
       
    Returns: 
       output (list): A list containing the detected objects
    '''
    w, h = image.size
    inx = 0
    output = []
    # Loop over the list of detected objects
    for res in megadetector_result['detections']:
        print(res)
        # Only process object of the category 1(animals)
        if res['category'] == '1':
            # Extract the bounding box in relative coordinates.
            xi, yi, wi, hi = res['bbox']
            # Convert the bounding box to pixel coordinates
            xi, yi, wi, hi = [int(xi*w), int(yi*h), int(wi*w), int(hi*h)]
            # Get the difference between width and height
            ds = abs(wi - hi)
            
            # Transform the bounding box to a square
            if wi > hi:
                hi = wi
                yi = yi - ds / 2
            else:
                wi = hi
                xi = xi - ds / 2

            box = [xi, yi, xi+wi, yi+hi]
            # Define the output file name, replacing the input forder by the output folder
            new_file = new_file = megadetector_result['file'].replace(input_folder, output_subfolder)
            if not os.path.exists(os.path.dirname(new_file)):
                os.makedirs(os.path.dirname(new_file))
            image2 = image.copy()
            # Crop the image and save it to the corresponding place.
            cropped = image2.crop(box).resize(size=(244, 244))
            output.append(cropped)
            cropped.save(new_file.replace('.JPG', f'_{inx}.JPG'))

            inx += 1 
    
    if inx == 0:
        print('Animals were not detected in this picture');
    
    return output


def plot_detections(
    image: ImageOps,
    detections: List,
    megadetector_result: Dict
):
    '''Plot the image with bounding boxes and the cropped images at its side.

    Args:
       image (ImageOps): A PIL image
       detections (dict): The MegaDetector output
       input_subfolder (str): A folder name to find the images
       output_subfolder (str): A folder name to save the cropped images
       
    Returns: 
        result: A list containing the detected objects
    '''
    number_of_detections = len(detections)
    if number_of_detections > 0:
        fig, axs = plt.subplots(
            1, number_of_detections + 1,
            figsize=(14,7),
            gridspec_kw={'width_ratios': [3] + [1] * number_of_detections}
        )
        axs[0].imshow(draw_bounding_box(image, megadetector_result))
        axs[0].set_title("original")
        axs[0].axis("off")
        for index, detection in enumerate(detections):
            axs[index + 1].imshow(detection)    
            axs[index + 1].set_title(f"crop_{index}")
            axs[index + 1].axis("off")
    else:
        plt.imshow(image)
        plt.axis("off")


def preprocess_dataset(
    root_dir: str,
    megadetector: PTDetector,
    cropper: Callable,
    detection_threshold: float = 0.6,
    n_samples: int = 6
):
    '''Crop the images and store the results.
    The function has an upper limit to process images, for demonstration purposes.

    Args:
       root_dir (str): Directory with the images
       megadetector (PTDetector): The MegaDetector model
       cropper (Callable): A function to crop and save the image
       detection_threshold (float): The MegaDetector output
       n_samples (int): Upper limit of images to process
    '''
    all_filenames = glob.iglob(f'{root_dir}/**/*.JPG', recursive=True)
    for count, im_file in enumerate(all_filenames):

        # We will only crop a few images.
        if count >= n_samples:
            break

        image = viz_utils.load_image(im_file)

        res = megadetector.generate_detections_one_image(image, im_file, detection_threshold=detection_threshold)
        print(res)
        output_folder = './data_crops/train'

        # Save crop to folder
        cropper(image, res, root_dir, output_folder)


def plot_cropped_images():
    '''Plots cropped images in a 3x3 matrix.'''
    all_cropped_images = list(glob.iglob(f'./data_crops/**/**/*.JPG', recursive=True))
    plt.figure(figsize=(15, 15))

    for loc_index in range(9):
        cropped_im_file = choice(all_cropped_images)
        cropped_image = viz_utils.load_image(cropped_im_file)
        ax = plt.subplot(3, 3, loc_index + 1)
        plt.imshow(cropped_image)
        plt.title(cropped_im_file.split('/')[3], fontsize=14)
        plt.axis("off")
    plt.show()   


# Functions used in the second design phase notebook

def ignore_tf_warning_messages():
    '''Ignore tf warning messages.'''
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    logging.getLogger("tensorflow").setLevel(logging.ERROR)
    tf.autograph.set_verbosity(0)


def load_data(
    image_dir: str,
    batch_size: int,
    image_size: Tuple[int, int],
    seed: int
) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
    '''Load the images as TensorFlow datasets.
    
    Args:
       image_dir (str): Directory with the images
       batch_size (int): The batch size
       image_size (tuple): A tuple of height and width of the images
       seed (int): Random seed
       
    Returns:
       train_ds (tf.data.Dataset): A TensorFlow dataset of train images
       val_ds (tf.data.Dataset): A TensorFlow dataset of validation images
       test_ds (tf.data.Dataset): A TensorFlow dataset of test images
    '''
    # 
    tf.random.set_seed(seed)

    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        f'{image_dir}/train',
        labels="inferred",
        label_mode="int",
        batch_size=batch_size,
        image_size=image_size,
        seed=seed,
        crop_to_aspect_ratio=True  # Preserve aspect ratio of the original images
    )

    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        f'{image_dir}/validation',
        labels="inferred",
        label_mode="int",
        batch_size=batch_size,
        image_size=image_size,
        seed=seed,
        crop_to_aspect_ratio=True
    )
    
    test_ds = tf.keras.preprocessing.image_dataset_from_directory(
        f'{image_dir}/test',
        labels="inferred",
        label_mode="int",
        batch_size=batch_size,
        image_size=image_size,
        seed=seed,
        crop_to_aspect_ratio=True
    )
    
    return train_ds, val_ds, test_ds


def count_examples_per_class(
    ds: tf.data.Dataset,
    label2cat: Dict[int, str],
    cat2label: Dict[str, int]
) -> Dict [str, int]:
    '''Count the number of data examples for each class.
    
    Args:
       ds (tf.data.Dataset): A tensorflow dataset
       label2cat (dict): Mapping from labels to categories
       cat2label (dict): Mapping from categories to labels
       
    Returns:
       values (dict): A dictionary mapping category names to their count

    '''
    values = {}
    for cat in cat2label:
        values[cat] = 0
    for x, y in ds:
        for y_i in y:
            values[label2cat[y_i.numpy()]] += 1
    return values


def plot_histograms_of_data(
    count_original: Dict [str, int],
    count_resampled: Dict [str, int],
    cat2label: Dict[str, int],
    label2cat: Dict[int, str]
):
    '''Plot the histograms of data counts.
    
    Args:
       count_original (dict): A dictionary mapping category names to their count
       count_resampled (dict): A dictionary mapping category names to their count
       cat2label (dict): Mapping from categories to labels
       label2cat (dict): Mapping from labels to categories
    '''    
    labels = list(cat2label.keys())

    plt.rcParams["figure.figsize"] = [18, 5]
    fig, (ax1, ax2) = plt.subplots(1, 2)
    labels = list(label2cat.values())
    ax1.bar(labels, [count_original[label] for label in labels])
    ax1.set_xticklabels(labels, rotation=90, fontsize=12)
    ax1.set_title('Original', fontsize=20)

    ax2.bar(labels, [count_resampled[label] for label in labels])
    ax2.set_xticklabels(labels, rotation=90, fontsize=12)
    ax2.set_title('Resampled', fontsize=20)

    plt.show()

    
def plot_single_image(
    image: np.ndarray,
    title: str
):
    '''Plot one image.
    
    Args:
       image (np.ndarray): An image to be plotted
       title (str): Title of the plot
    '''    
    plt.figure()
    plt.imshow(image)
    plt.title(title)
    plt.axis("off")
    plt.show()


def data_aug_flip(image_single_batch: tf.Tensor):
    '''Interactive function to illustrate flipping of images.
    
    Args:
       image_single_batch (tf.Tensor): An image to be displayed
    '''    
    image_single_batch = tf.expand_dims(image_single_batch, 0)
    img = ImageOps.fromarray(image_single_batch[0].numpy())

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
        plt.show()

    random_flip_widget = widgets.RadioButtons(
        options=['horizontal', 'vertical', 'horizontal_and_vertical'],
        value='horizontal',
        layout={'width': 'max-content'}, # If the items' names are long
        description='Random Flip',
        disabled=False,
        style = {'description_width': 'initial'},
    )
    
    interact(_data_aug_flip, random_flip = random_flip_widget)
    
    
def data_aug_zoom(image_single_batch: tf.Tensor):
    '''Interactive function to illustrate zooming images.
    
    Args:
       image_single_batch (tf.Tensor): An image to be displayed
    '''
    image_single_batch = tf.expand_dims(image_single_batch, 0)
    img = ImageOps.fromarray(image_single_batch[0].numpy())

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
        plt.show()

    zoom_widget = widgets.FloatSlider(
        value=1,
        min=1,
        max=2,
        step=0.1,
        description='Zoom: ',
        disabled=False,
        continuous_update=False,
        orientation='horizontal',
        readout=True,
        readout_format='.1f',
        style = {'description_width': 'initial'}
    )
    interact(_data_aug_zoom, zoom_factor = zoom_widget)


def data_aug_rot(image_single_batch: tf.Tensor):
    '''Interactive function to illustrate rotating images.
    
    Args:
       image_single_batch (tf.Tensor): An image to be displayed
    '''
    image_single_batch = tf.expand_dims(image_single_batch, 0)
    img = ImageOps.fromarray(image_single_batch[0].numpy())
    
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
        plt.show()

    angle_widget = widgets.FloatSlider(
        value=0,
        min=-45,
        max=45,
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
    
    
def data_aug_contrast(image_single_batch: tf.Tensor):
    '''Interactive function to illustrate contrasting images.
    
    Args:
       image_single_batch (tf.Tensor): An image to be displayed
    '''
    image_single_batch = tf.expand_dims(image_single_batch, 0)
    img = ImageOps.fromarray(image_single_batch[0].numpy())
    enhancer = ImageEnhance.Contrast(img)

    def _data_aug_contrast(contrast_factor):
        plt.figure(figsize=(10, 10))
        ax = plt.subplot(1, 2, 1)
        ax.imshow(img)
        ax.set_title('Original')
        plt.axis("off")
        
        img2 = enhancer.enhance(contrast_factor)
        ax = plt.subplot(1, 2, 2)
        ax.imshow(img2)
        ax.set_title('Contrasted')
        plt.axis("off")
        plt.show()

    contrast_widget = widgets.FloatSlider(
        value=1,
        min=0.5,
        max=1.5,
        step=0.2,
        description='Contrast factor: ',
        disabled=False,
        continuous_update=False,
        orientation='horizontal',
        readout=True,
        readout_format='.1f',
        style = {'description_width': 'initial'},
    )
    interact(_data_aug_contrast, contrast_factor = contrast_widget)

    
def data_aug_random(image_single_batch: tf.Tensor):
    '''Interactive function to illustrate combined data augmentation steps.
    
    Args:
       image_single_batch (tf.Tensor): An image to be displayed
    '''
    image_single_batch = tf.expand_dims(image_single_batch, 0)
    img = ImageOps.fromarray(image_single_batch[0].numpy())
    
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


def get_test_imgs(test_dir: str) -> pd.core.frame.DataFrame:
    '''Reads trough the folder structure and creates a dataset containing test images metadata.
    
    Args:
        test_dir (str): Directory of test images
       
    Returns:
        test_imgs (pd.core.frame.DataFrame): Dataframe with metadata
    '''
    test_imgs = []
    for animal in os.listdir(test_dir):
        files = [os.path.join(test_dir, animal, filename) for filename in os.listdir(f'{test_dir}/{animal}')]
        for t in files:
            test_imgs.append([animal, t])
    test_imgs = pd.DataFrame(test_imgs, columns=['animal', 'filename'])
    return test_imgs


def pick_img_and_plot_predictions(
    test_img_df: pd.core.frame.DataFrame,
    model: tf.keras.Model,
    label2cat: Dict[int, str],
    cat2label: Dict[str, int],
    image_size: Tuple[int, int]
):
    '''Plot the histograms of data counts.
    
    Args:
       test_img_df (pd.core.frame.DataFrame): A dataframe with metadata of test images
       model (tf.keras.Model): A model to predict the classes on the images
       label2cat (dict): Mapping from labels to categories
       cat2label (dict): Mapping from categories to labels
       image_size (tuple): A tuple of height and width of the images
    '''   
    choose_animal_widget = widgets.Dropdown(options=cat2label.keys())
    # Figure out how to make each axis a square/same size as original IMAGE_SIZE
    
    def _pick_img_and_plot_predictions(animal):
        """
            Pick random filename given the animal chosen on the dropdown
            Then, plots grad cam for that filename
        """
        
        filename = test_img_df[test_img_df.animal==animal].sample(1)["filename"].values[0] 
        fig, axes = plt.subplots(
            1, 2,
            figsize=(15, 6),
            gridspec_kw={'width_ratios': [1, 1], 'wspace': 0.5}
        )

        img_array = get_img_array(filename, image_size)
        # In the first box, plot the original image
        plot_original_image(img_array, axes[0])

        # In the third box, plot the predictions
        preds = model.predict(img_array)
        
        predictions = get_probabilities_for_img_array(preds, label2cat) 
        plot_predictions(predictions, animal, ax=axes[1], topn=3)

        # Title
        plt.suptitle(f'True: {animal}; Prediction: {predictions.loc[0, "category"]}', size=16)

    interact(_pick_img_and_plot_predictions, animal=choose_animal_widget)    


def get_img_array(
    img_path: str,
    image_size: Tuple[int, int]
) -> np.ndarray:
    '''Loads image and returns it as a batch of a single image.
    
    Args:
        img_path (str): Path to the image
        image_size (tuple): A tuple of height and width of the image

    Returns:
        array (np.ndarray): Image in a 4D array
    '''
    img = tf.keras.preprocessing.image.load_img(img_path, target_size=image_size)
    # `array` is a float32 Numpy array of shape (image_size[0], image_size[1], 3)
    array = tf.keras.preprocessing.image.img_to_array(img)
    # Add a dimension to transform our array into a "batch" of size
    # (1, image_size[0], image_size[1], 3)
    array = np.expand_dims(array, axis=0)
    
    return array


def plot_original_image(
    img_array: np.ndarray,
    ax: plt.Axes
):
    '''Plots an image to a specified Axes.
    
    Args:
        img_array (np.ndarray): Image stored in an array
        ax (plt.Axes): Axes on the plot where to plot the image
    '''
    ax.imshow(img_array[0].astype("uint8"))
    ax.axis('off')
    ax.set_title('Original image', fontsize=16)


def get_probabilities_for_img_array(
    preds: np.ndarray,
    label2cat: Dict[int, str]
) -> pd.core.frame.DataFrame:
    '''Given predictions array, create a dataframe with names and predictions of the top classes.
    
    Args:
        preds (np.ndarray): Predictions for classes
        label2cat (dict): Mapping from labels to categories

    Returns:
        predictions (pd.core.frame.DataFrame): Dataframe with predictions for the top classes
    '''
    predictions = []
    if type(label2cat) is dict:
        for p, l in zip(preds[0], label2cat):
            predictions.append({
              "category": label2cat[l],
              "probability": p
            })
    else:
        preds_list = label2cat(preds, top=20)
        for prediction in preds_list[0]:
            predictions.append({
              "category": prediction[1],
              "probability": prediction[2]
            })
    
    predictions = pd.DataFrame(predictions)
    predictions.sort_values(by="probability", ascending=False, inplace=True)
    predictions.reset_index(inplace=True, drop=True)
    return predictions


def plot_predictions(
    predictions: pd.core.frame.DataFrame,
    true_animal: str,
    ax: plt.Axes,
    topn: Optional[int] = None
):
    '''Given a df of top predictions, plots the top n most likely classes.
    
    Args:
        predictions (pd.core.frame.DataFrame): Dataframe with predictions for the top classes
        true_animal (str): True label of the image
        ax (plt.Axes): Axes on the plot where to plot the image
        topn (int): Number of top predictions to display

    Returns:
        predictions (pd.core.frame.DataFrame): Dataframe with predictions for the top classes
    '''
    if topn is None:
        data = predictions
        topn = len(predictions)
    else: data = predictions.head(topn)
    
    palette = ['green' if predictions['category'].values[i] == true_animal else 'red' for i in range(3)]

    f = sns.barplot(x='probability',
                    y='category',
                    data=data, 
                    palette=palette,
                    ax=ax
       )
    sns.set_style(style='white')
    f.grid(False)
    f.spines['top'].set_visible(False)
    f.spines['right'].set_visible(False)
    f.spines['bottom'].set_visible(False)
    f.spines['left'].set_visible(False)
    f.set_title(f'Top {topn} Predictions:', fontsize=16)
    f.set_xlabel('Probability', fontsize=14)
    f.set_ylabel('Category', fontsize=14)

    
def get_transfer_model(
    model_to_transfer: tf.keras.Model,
    num_classes: int,
    img_height: int,
    img_width: int,
    num_channels: int=3
) -> tf.keras.Model:
    '''Create a model based on a base model
    
    Args:
        model_to_transfer (tf.keras.Model): tf keras model to use as a base
        num_classes (int): Number of classes for classification
        img_height (int): Input image height
        img_width (int): Input image width
        num_channels (int): Number of channels in the image

    Returns:
        model (tf.keras.Model): Updated model with a new top.
    '''
    inputs = tf.keras.Input(shape=(img_height, img_width, num_channels))
    x = image_mutation(input_shape=(img_height, img_width, num_channels))(inputs)
    x = nasnet.preprocess_input(x)
    
    # instantiate the model without the top (classifier at the end)
    # transfer = model_to_transfer(include_top=False)
    
    # freeze layer weights in the transfer model to speed up training
    for layer in model_to_transfer.layers:
        layer.trainable = False
        if isinstance(layer, Model):
            freeze(layer)
        
    x = model_to_transfer(x)
    x = GlobalMaxPooling2D(name="pooling")(x)
    x = Dropout(0.1, name="dropout_1")(x)
    x = Dense(256, activation="relu", name="dense2")(x)
    x = Dropout(0.1, name="dropout_3")(x)
    outputs = Dense(num_classes, activation="softmax", name="classifer")(x)

    model = Model(inputs=inputs, outputs=outputs)
    
    model.compile(
      optimizer=tf.keras.optimizers.Adam(5e-6),
      loss=tf.keras.losses.SparseCategoricalCrossentropy(),
      metrics=["accuracy", sparse_top_k_categorical_accuracy],
    )

    return model    
  

class image_mutation(tf.keras.layers.Layer):
    '''Apply a set of image mutations to a given batch of images. 
    The transformation only apply during training
    '''
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.data_augmentation = tf.keras.Sequential(
              [
                  tf.keras.layers.RandomFlip("horizontal_and_vertical", input_shape=kwargs['input_shape']),
                  tf.keras.layers.RandomRotation(0.2),
                  tf.keras.layers.RandomZoom(0.1),
                  tf.keras.layers.RandomContrast(0.4)
              ]
          )
        self.trainable=False
    def call(self, inputs, training=None):
        if training:
            return self.data_augmentation(inputs)
        else:
            return inputs
    
    
def plot_cm(
    y_true: Iterable,
    y_pred: Iterable,
    label2cat: Dict[int, str]
):
    '''Plot confusion matrix
    
    Args:
        y_true (Iterable): True labels
        y_pred (Iterable): Predicted labels
        label2cat (dict): Mapping from labels to categories
    '''
    def _plot_cm(true_categories, predicted_categories, normalize=True):
        # Specify the labels we want on our ticks
        labels = label2cat.values()

        # Create confusion matrix
        cm = confusion_matrix(true_categories, predicted_categories)

        if normalize: 
            # Normalize
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            # Remove divide by 0 nans
            cm = np.nan_to_num(cm)
            # Set number format as 2 decimal points
            fmt='.2f'
        else:    
            # Set the number format as whole numbers
            fmt='d'

        plt.figure(figsize=(10,10))
        sns.heatmap(cm, annot=True, fmt=fmt, xticklabels=labels, yticklabels=labels, cmap='Purples')
        plt.xlabel('Predicted Labels', size=14)
        plt.ylabel('True Labels', size=14)
        plt.title('Confusion Matrix', size=16)
        plt.show()

    cm_normalize_widget = widgets.RadioButtons(
        value=True,
        options=[True, False],
        disabled=False,
        description='Normalize?'
    )
    
    interact(_plot_cm, 
         true_categories=fixed(y_true),
         predicted_categories=fixed(y_pred),
         normalize = cm_normalize_widget);    

    
def plot_training_history(filename: str):
    '''Plot the training history of the model
    
    Args:
        filename (str): Path to the file where training history is stored.
    '''
    # open a file, where you stored the pickled data
    with open (filename, 'rb') as file:
        data=pickle.load(file)

    history_training = pd.DataFrame(data=data)

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
    fig.update_xaxes(title_text="Number of epochs", row=1, col=1)
    fig.update_xaxes(title_text="Number of epochs", row=1, col=2)

    # Update yaxis properties
    fig.update_yaxes(title_text="Accuracy", row=1, col=1)
    fig.update_yaxes(title_text="Loss", row=1, col=2)

    # Update title and height
    fig.update_layout(title_text="Training metrics", height=400)
    fig.show()


def resample_data(
    input_dir: str, output_dir: str,
    train_ds: tf.data.Dataset,
    n_classes: int,
    n_samples_x_class: int
):
    '''Resamples he data using oversampling and undersampling
    to create a balanced dataset.
    
    Args:
        input_dir (str): Input data directory
        output_dir (str):  Output data directory
        train_ds (tf.data.Dataset): Training dataset
        n_classes (int): Number of classes
        n_samples_x_class (int): Desired number of samples per class after resampling
    '''
    shutil.rmtree(output_dir)
    os.mkdir(output_dir)
    os.mkdir(os.path.join(output_dir, 'train'))
    os.mkdir(os.path.join(output_dir, 'test'))
    os.mkdir(os.path.join(output_dir, 'validation'))
    
    # Get the labels and categories
    label2cat = {i:category for i, category in enumerate(sorted(next(os.walk(f'{input_dir}/train'))[1]))}
    cat2label = {v:k for k,v in label2cat.items()}

    count_original = count_examples_per_class(train_ds, label2cat, cat2label)
    
    # First n classes
    labels = np.argsort([count_original[label] for label in cat2label.keys()])[::-1][:n_classes]
    #Create structure at train
    
    for label in labels:
        shutil.copytree(os.path.join(input_dir, 'test', label2cat[label]), os.path.join(output_dir, 'test', label2cat[label]))
        shutil.copytree(os.path.join(input_dir, 'validation', label2cat[label]), os.path.join(output_dir, 'validation', label2cat[label]))
            
        if count_original[label2cat[label]] >= n_samples_x_class:
            train_path = os.path.join(output_dir, 'train', label2cat[label])
            if not os.path.exists(train_path):
               # Create a new directory because it does not exist
               os.makedirs(train_path)
            #Copy with Sub-Sampling
            input_path = os.path.join(input_dir, 'train', label2cat[label],'*.JPG')
            files = random.sample(glob.glob(input_path), n_samples_x_class)
            for file in files:
                shutil.copyfile(file, file.replace(input_dir, output_dir))
        else:      
            #Copy with Overs-Sampling
            shutil.copytree(os.path.join(input_dir, 'train', label2cat[label]), os.path.join(output_dir, 'train', label2cat[label]))
            missing = n_samples_x_class - count_original[label2cat[label]]
            
            input_path = os.path.join(input_dir, 'train', label2cat[label],'*.JPG')
            files = np.random.choice(glob.glob(input_path), missing)
            index = 0
            for file in files:
                shutil.copyfile(file, file.replace(input_dir, output_dir).replace('.JPG', '_' +str(index)+'.JPG'))
                index += 1
                