# Datasheet: *Hurricane Harvey data* Lab 1

Author: DeepLearning.AI (DLAI)

Files:
	14000 .jpeg satellite images

## Motivation

The dataset consists of 14000 .jpeg satellite images of the region impacted by the hurricane Harvey in USA. It is a modified version of the dataset taken from  [IEEE Data Port](https://ieee-dataport.org/open-access/detecting-damaged-buildings-post-hurricane-satellite-imagery-based-customized). 

The dataset contains images before and after the hurricane. Some locations include images before and after, other locations include only the image before or the image after.

## Composition

The original data set contains data in 4 folders: 'train_another', 'validation_another', 'test_another' and 'test'. Since 'test_another' is not balanced, we do not use it here, but we use the 'test' folder, so 'train_another' is not present in this dataset. Additionally we remove the suffix 'another' from the train and validation folders for clarity.
the 'train folder includes 10000 images, the 'validation' and 'test' folders include 2000 images each. Each subset of data is evenly balanced between the classes.

Each folder has two subfolders, which represent classes: 'damage' and 'no_damage'. We have renamed the 'damage' folders to 'visible_damage'.

The name of each file represents the geographical coordinates of the image, separated by a comma.