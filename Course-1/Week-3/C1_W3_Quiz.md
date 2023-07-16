**1.** What were the initial indications from your exploration of the data that suggested that Al might add value in addressing the problem of filling in missing values in the data? Select all that apply.
- [x] There appear to be spatial patterns in the data, namely that individual station measurements are consistently higher or lower depending on location.
- [x] There appear to be temporal patterns in the data, like daily or weekly patterns that repeat.
- [x] There appear to be correlations between many of the individual pollutant measurements.

**2.** What are some possible approaches to replace the m issing values in the data? Select all that apply.
- [x] Use an algorithm, like a neural network, to estimate the missing value based on information from other sensor measurements as well as things like location and time of day.
- [x] Copy the last available measurement from the sensor station that is currently offline.
- [x] Copy the current measurement from the closest sensor station that is online.

**3.** When it comes to designing a solution for a problem where you think Al might add value, what is a good general approach?
- [x] Start with a simple method to establish a baseline. Then try more complex algorithms and compare them with your baseline results.
- [ ] Do some research on what Al model would best address the problem you're working on. Start with this model, and then try modifying various parameters to see if you can get an improved result.

**4.** Why can a more complex method, a neural network in this case, for estimating the missing values outperform the simplest methods like copying the last recorded value or using the nearest neighbor method? Select all that apply.
- [x] The more complex method, a neural network in this case, can learn from patterns in the data (correlations, temporal and spatial patterns).
- [ ] More complex models always perform better than simple techniques.
- [x] The more complex method, a neural network in this case, can capture nonlinear relationships between features in the dataset.

**5.** What are some of the possible inherent challenges in accurately estimating pollution levels in between the sensors? Select all that apply.
- [x] Pollution levels between the sensors will depend on many factors, for example, which way the wind is blowing, making a uniform distance weighting scheme limited in its practicality.
- [x] You don't actually have any "ground truth' measurements for pollution levels at locations where there are no sensors so any model you adopt can only be a rough estimate.
- [ ] There are not enough sensors to make a meaningful estimate of pollution in between the sensors.

**6.** What would be one example of a risk for doing harm with an air quality monitoring project like this?
- [x] Your product is meant to inform the public about health risks due to poor air quality and if it malfunctioned it could cause people to be unaware of the risks they face. 
- [ ] The data associated with this project is proprietary information and could pose a security risk to the city of Bogota such that inadvertently releasing the data could cause harm.
- [ ] The data for this project contains personally identifiable information such that publishing or insecurely storing the data could cause risk to individuals.

**7.** What metric do you use in the labs to assess the performance of various models (Design phase lab videos)?

- [ ] K Nearest Neighbors (KNN).
- [ ] Standard deviation.
- [ ] A neural network.
- [x] Mean absolute error (MAE).
- [ ] Mean squared error (MSE).

**8.** What can you say about the interpolated data between the sensor stations when using one nearest neighbor vs. three nearest neighbors? Select all that apply.
- [ ] The predictions are equally good when using one or three nearest neighbors.
- [x] When using three nearest neighbors, the interpolation on the map looks smoother.
- [x] When using three nearest neighbors, the MAE is lower, thus the predictions can be expected to be better on average.
- [ ] Distance weighting with one nearest neighbor produces a lower MAE value than distance weighting with three nearest neighbor.

**9.** What are some of the properties of artificial neural networks? Select all that apply.
- [x] An artificial neural network is a computation machine that can take in a collection of inputs, run a computation, and generate an output.
- [x] A neural network is a particular kind of machine learning model.
- [x] An artificial neural network is made up of layers of so-called artificial neurons.
- [x] Artificial neural networks can generally model more complex functions of data than simple linear models.

**10.** Which of the sentences best describes the inverse distance weighting scheme?
- [ ] You weight the measurements based on the distance from the point of interest, such that the measurements that are closer have a lower weight than the measurements further away.
- [ ] You weight the measurements based on the square of the distance to the center of the coordinate system.
- [x] You weight the measurements based on the distance from the point of interest, such that the measurements that are closer have a higher weight than the measurements further away.
