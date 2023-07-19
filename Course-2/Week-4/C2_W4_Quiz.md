**1.** Which of the following statements are true about convolutional neural networks (CNNs)? Select all that apply.
- [x] CNNs can be trained using supervised learning techniques by showing them many examples of images and labels.
- [ ] CNNs are always the best choice when it comes to algorithms for working with image data.
- [x] They are Al models made of a set of interconnected layers that are optimized for learning patterns in images.

**2.** True or false: A model that has been pre-trained for the task you hope to work on might perform sufficiently well without any further training
- [x] True.
- [ ] False.

**3.** Which of the following statements are true about fine-tuning models? Select all that apply.
- [x] In machine learning, fine-tuning a model means taking a model that was already pre-trained for some task and performing additional training on the dataset you're interested in.
- [ ] Fine-tuning a pre-trained model will always provide a better result than training your model from scratch.
- [ ] When fine-tuning, you can only train the last few layers of your neural network.
- [x] Fine-tuning can often achieve results that would require much more data if you were training from scratch.

**4.** Which sentences about MegaDetector are correct? Select all that apply.
- [ ] MegaDetector was trained on a large corpus of images and can thus distinguish between 1000 classes of different animals.
- [x] MegaDetector can distinguish between three different classes: animal, vehicle and person, and identify multiple objects and / or classes of objects in the same image.
- [x] MegaDetector can tell you the location of an object in the image by identifying a bounding box that goes around the object.

**5.** Why does it make sense to use MegaDetector to crop the animals in the images before attempting to classify them? Select all that apply.
- [x] Cropping helps because it allows the classifier model to only focus on the region of interest and the majority of the background is eliminated.
- [ ] Cropping out detected objects guarantees that the full object of interest, whether animal, person, or vehicle, is entirely visible.
- [x] A single image might contain multiple objects and so first separating and cropping them out helps narrow down the classification task.

**6.** True or false: The purpose of data augmentation is to produce additional labeled examples for training that are similar but not identical to the original examples in your dataset
- [ ] False.
- [x] True.

**7.** What are some of the techniques you can use for image data augmentation? Select all that apply.
- [x] Modifying brightness and contrast of the images
- [x] Applying a zoom factor to images.
- [x] Flipping and rotating the image.

**8.** With the NASNet pre-trained model that you fine-tuned for classification of animals, how would you characterize the model's performance before fine-tuning?
- [ ] The model performed relatively well but in some cases misclassified the animals as other objects, like a bridge or a boat.
- [x] The model provided predictions that were wrong in every case, misclassifying animals for things like boats, trains, and lemons, sometimes even with high confidence.
- [ ] The model did not perform particularly well, but in some cases it accurately classified animals.

**9.** After fine-tuning the NASNet model for classification, what were you able to observe within the confusion matrix?
- [x] The percentage of times your model correctly classified each type of animal.
- [x] The percentage of times your model incorrectly classified each animal as another type of animal.
- [ ] The percentage of times your model classified an image as containing something other than the eleven classes of animals it was trained to recognize.

**10.** True or false: With your final image processing pipeline you could, at least in principle, upload an image that contained a zebra, a black backed jackal, a kudu, and a kori bustard, as well as several people and a vehicle, and get an output with bounding boxes drawn around the animals, people, and vehicles, and each of the animals identified as to what it was.
- [ ] False.
- [x] True.
