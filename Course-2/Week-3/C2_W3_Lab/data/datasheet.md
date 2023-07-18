# Datasheet: *Karoo camera trap data* Lab 1

Author: DeepLearning.AI (DLAI)

Files:
	6200 .JPG files

## Motivation

The dataset consists of 6200 .JPG images from camera traps in Karoo national park. It is a cleaned-up version of the Snapshot Karoo dataset, which is part of the [Lila BC project](https://lila.science/datasets/snapshot-karoo). The full data set contains 14889 sequences of camera trap images, totaling 38074 images.

The dataset has been annotated thanks to the contributions of volunteers using [this tool](https://www.zooniverse.org/projects/shuebner729/snapshot-karoo/classify). The images are taken as a sequence when camera trap is triggered and are also annotated at sequence level, thus sometimes the pictures are annotated as containing an animal even if the animal is not visible in the image, because an animal was present in another image of the sequence.

The full quality Karoo dataset is 42 GB. For the purpose of this notebook, just the images containing animals were selected (the images that are labelled as empty or having vehicles were deleted). With this the size of the dataset is reduced to 6 GB. Furthermore, reducing the quality of each picture to around 100Kb, the dataset size is reduced to around 600 MB, which is feasible for this notebook. 

## Composition

The data consists of 6200 images with standardized naming. Each image is named as KAR_S1_XXX_R1_IMAGYYYY.JPG, where KAR_S1 means "Karoo, season 1" (same for all of the images in the dataset, XXX is a three character code for the location of the camera trap, R1 means "repetition 1" (all of the images in the dataset have R1), and IMAGYYYY stands for the unique number of the image.

The data has been taken at the following 15 locations:
'B03', 'B02', 'E01', 'C02', 'A01', 'B01', 'C03', 'A02', 'D03', 'D04', 'D01', 'E03', 'C04', 'F02', 'E02'

and contains the following classes (animals):
'baboon', 'birdother', 'birdsofprey', 'bustardkori', 'bustardludwigs', 'caracal', 'duiker', 'eland', 'foxbateared', 'foxcape', 'gemsbokoryx', 'hare', 'hartebeestred', 'hyenabrown', 'jackalblackbacked', 'klipspringer', 'kudu', 'lionfemale', 'lionmale', 'meerkatsuricate', 'mongoosesmallcapegrey', 'mongooseyellow', 'monkeyvervet', 'ostrich', 'porcupine', 'rabbitriverine', 'reedbuckmountain', 'reptilesamphibians', 'rhebokgrey', 'rhinocerosblack', 'springbok', 'steenbok', 'tortoise', 'wildebeestblue', 'zebraburchells', 'zebramountain'

The images are sorted in folders based on the class they belong to.

The data has a high imbalance between classes, as well as locations.
