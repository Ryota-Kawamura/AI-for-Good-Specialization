import io
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras.applications import nasnet
from tensorflow.keras.layers import Dense, Dropout, GlobalMaxPooling2D, Input
from detection.pytorch_detector import PTDetector

from PIL import Image, ImageDraw, ImageFont

# Libraries for interactive components
import ipywidgets as widgets
from IPython.display import display

import visualization.visualization_utils as viz_utils

from ipyfilechooser import FileChooser
from typing import Callable, List, Tuple, Dict, Optional, Any, Iterable, Union


def get_labels() -> Dict[int, str]:
    '''
    Returns a mapping from class numbers to class names.
    
    Returns:
        _ (dict): Mapping from labels to categories
    '''
    return {0: 'baboon',
             1: 'bustardkori',
             2: 'duiker',
             3: 'eland',
             4: 'gemsbokoryx',
             5: 'hartebeestred',
             6: 'jackalblackbacked',
             7: 'kudu',
             8: 'springbok',
             9: 'steenbok',
             10: 'zebramountain'}


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
    inputs = keras.Input(shape=(img_height, img_width, num_channels))
    x = nasnet.preprocess_input(inputs)
    
    # instantiate the model without the top (classifier at the end)
    # transfer = model_to_transfer(include_top=False)
    
    # freeze layer weights in the transfer model to speed up training
    for layer in model_to_transfer.layers:
        layer.trainable = False
        if isinstance(layer, Model):
            freeze(layer)
        
    x = model_to_transfer(x)
    x = GlobalMaxPooling2D(name="pooling")(x)
    x = Dropout(0.2, name="dropout_1")(x)
    x = Dense(256, activation="relu", name="dense")(x)
    x = Dropout(0.2, name="dropout_2")(x)
    outputs = Dense(num_classes, activation="softmax", name="classifer")(x)

    model = Model(inputs=inputs, outputs=outputs)

    return model


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
    ax.set_title('Original image')

    
def get_probabilities_for_img_array(
    img_array: np.ndarray,
    model: tf.keras.Model,
    label2cat: Dict[int, str]
) -> pd.core.frame.DataFrame:
    '''Given an image and a model, create a dataframe with names and predictions of the top classes.
    
    Args:
        img_array (np.ndarray): Image
        model (tf.keras.Model): Model
        label2cat (dict): Mapping from labels to categories

    Returns:
        predictions (pd.core.frame.DataFrame): Dataframe with predictions for the top classes
    '''    
    # Predict the labels for a single image
    preds = model.predict(img_array)

    predictions = []
    # Format the result, for having each category plus its probablity into an array
    for p, l in zip(preds[0], label2cat):
        predictions.append({
          "category": label2cat[l],
          "probability": p
        })
    predictions = pd.DataFrame(predictions)
    predictions.sort_values(by="probability", ascending=False, inplace=True)
    predictions.reset_index(inplace=True, drop=True)
    return predictions


def crop_image(image: Image, result: Dict,) -> np.ndarray:
    '''Crop the given image using the result from MegaDetector. Save all the crops into the given
    subfolder

    Args:
       image (Image): A PIL image
       megadetector_result (dict): The MegaDetector output

    Returns: 
       cropped (np.array): An array of cropped images
    '''
    # Get the width and height of the image(in pixels)
    w, h = image.size
    # You will store here each patch of the image containing an object
    cropped = []
    # Loop over all the detected objects
    for res in result['detections']:
        # Get the bounding box coordinates
        xi, yi, wi, hi = res['bbox']
        # Transform the MegaDetector output to pixels
        xi, yi, wi, hi = [int(xi * w), int(yi * h), int(wi * w), int(hi * h)]
        
        # Make the patches square. 
        ds = abs(wi - hi)

        if wi > hi:
            hi = wi
            yi = yi - ds / 2
        else:
            wi = hi
            xi = xi - ds / 2

        box = [xi, yi, xi + wi, yi + hi]
        # Crop the squared box of the object and append it to the result set
        cropped.append(np.array(image.crop(box).resize(size=(224, 224))))
    
    cropped = np.array(cropped)
    # Return an array containing all the patches of the images
    return cropped


def draw_bounding_box(
    sample_im_file: Image,
    megadetector: PTDetector,
    model: tf.keras.Model,
    label2cat: Dict[int, str]
) -> Image:
    ''' Draw a bounding on the given image, containing the first detected animal

    Args:
       sample_im_file (Image|str):  A PIL image or a path string
       megadetector (PTDetector): The MegaDetector model
       model (tf.keras.Model): The classification model
       label2cat (dict): Mapping from labels to categories
       
    Returns: 
       image2 (Image): Image with annotations
       image (Image): Image without annotations

    '''
    md_categories = {'1': 'Animal', '2': 'Human', '3': 'Vehicle'}
    md_colors = {'1': 'magenta', '2': 'red', '3': 'blue'}

    image = sample_im_file
    # If sample_im_file is a string load the image
    if type(sample_im_file)==str: 
        image = viz_utils.load_image(sample_im_file)
    # Use MegaDetector to detect the objects in the picture.
    detections = megadetector.generate_detections_one_image(image, sample_im_file, detection_threshold=0.4)
    # Get the regions of the image containing animals
    cropped_images = crop_image(image, detections)
    # Some font scaling depending on the image size
    font_size = int(image.size[1]*0.024)
    interlines = font_size + int(font_size * 0.15)
    FONT = ImageFont.truetype("DejaVuSans-Bold.ttf", size=font_size)
    
    # You will add the annotations into a copy of the image
    image2 = image.copy()

    # If there are animals detected on the picture...
    if len(cropped_images) > 0:
        # Loop over each detected object in the image
        for k in range(len(detections['detections'])):
            detection = detections['detections'][k]
            if detection['category'] == '1':
                # Use our animal classifier to classify each patch(region of interes)
                preds = get_probabilities_for_img_array(np.array([cropped_images[k]]), model, label2cat).values
                # Get the bounding box
                bbox = detection['bbox']
                # Get the width and height of the image(in pixels)
                w, h = image.size
                # Transform the bound box coordinates to pixels
                x1, y1 = (int(bbox[0] * w), int(bbox[1] * h))
                # Create a rectangle using the MegaDetector's output, transforming all coordinates to pixels
                shape = [(x1, y1), (int((bbox[0] + bbox[2]) * w), int((bbox[1] + bbox[3]) * h))]
                # Draw the rectange over the copy of the image
                img1 = ImageDraw.Draw(image2)  
                img1.rectangle(shape, outline ="magenta")
                
                # Add the legends. 
                # This control if the text appears inside or outside the box.
                offset = 0
                if y1 - interlines * 3 < 0:
                    offset = interlines * 3
                # Still adding the legends. We need to make some arithmetic here.
                img1.text((x1, y1 - interlines * 3 + offset), str(round(preds[0, 1], 2)) 
                          + ' : ' + preds[0, 0], fill=(0,255,0, 255), font=FONT)
                img1.text((x1, y1 - interlines * 2 + offset), str(round(preds[1, 1], 2)) 
                          + ' : ' + preds[1, 0], fill=(255, 165,0), font=FONT)
                img1.text((x1, y1 - interlines + offset), str(round(preds[2, 1], 2)) 
                          + ' : ' + preds[2, 0], fill=(255, 80, 10), font=FONT)
            else:
                bbox = detection['bbox']
                # Get the width and height of the image(in pixels)
                w, h = image.size
                # Transform the bound box coordinates to pixels
                x1, y1 = (int(bbox[0] * w), int(bbox[1] * h))
                # Create a rectangle using the MegaDetector's output, transforming all coordinates to pixels
                shape = [(x1, y1), (int((bbox[0] + bbox[2]) * w), int((bbox[1] + bbox[3]) * h))]
                # Draw the rectange over the copy of the image
                img1 = ImageDraw.Draw(image2)  
                img1.rectangle(shape, outline =md_colors[detection['category']])
                
                # Add the legends. 
                # This control if the text appears inside or outside the box.
                offset = 0
                if y1 - interlines < 0:
                    offset = interlines
                # Still adding the legends. We need to make some arithmetic here.
                img1.text((x1, y1 - interlines + offset), md_categories[detection['category']], fill=(0,255,0, 255), font=FONT)
                

        return image2
    else:
        print('Animals were not detected in this picture');
        return image

    
def animal_detection_on_server(
    display: display,
    megadetector: PTDetector,
    model: tf.keras.Model,
    label2cat: Dict[int, str]
) -> Dict:
    '''Display a file chooser widget that allows to select files that are in the server side
        
    Args:
        display (display): The widget display to render the ouputs
        megadetector (PTDetector): The MegaDetector model
        model (tf.keras.Model): The classification model
        label2cat (dict): Mapping from labels to categories
        
    Returns: 
       _ (dict): Dictionary of widgets to display
    '''
    fc = FileChooser('./sample_data/test/')

    main_display = widgets.Output()

    def animal_detection_on_image(chooser):
        with  main_display:
            main_display.clear_output()
            image = Image.open(chooser.selected)
            # This function will crop the image using MD, classify the animal, 
            # and then produce the picture that will be shown.
            result = draw_bounding_box(image, megadetector, model, label2cat)
            display(result)

    fc.register_callback(animal_detection_on_image)
    
    return {'fileChooser': fc, 'output': main_display}


def animal_detection_local(
    display: display,
    megadetector: PTDetector,
    model: tf.keras.Model,
    label2cat: Dict[int, str]
) -> Dict:
    '''Display a file chooser widget that allows to select files that are in the users computer.
        
    Args:
        display (display): The widget display to render the ouputs
        megadetector (PTDetector): The MegaDetector model
        model (tf.keras.Model): The classification model
        label2cat (dict): Mapping from labels to categories
        
    Returns: 
       _ (dict): Dictionary of widgets to displa
    '''
    
    uploader = widgets.FileUpload(
        accept='image/*',  # Accepted file extension e.g. '.txt', '.pdf', 'image/*', 'image/*,.pdf'
        multiple=False  # True to accept multiple files upload else False
    )

    main_display = widgets.Output()

    def on_upload_change(change):
        with  main_display:
            main_display.clear_output()
            keys = list(uploader.value.keys())
            print(keys)
            image = Image.open(io.BytesIO(uploader.value[keys[0]]['content']))
            result = draw_bounding_box(image, megadetector, model, label2cat)
            display(result)

    uploader.observe(on_upload_change, names='_counter')
    
    return {'fileUpload': uploader, 'output': main_display}