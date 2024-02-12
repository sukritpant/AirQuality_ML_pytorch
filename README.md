# Air Quality Based Humidity and Temperature #

The project analyzes the Air Quality Data and measures the impact of various air molecules to the humidity and temperature for the given timeframe  as part of my Integrify Academy Machine Learning and Deep Learning project. The model is trained on the data to test if the content of molecular particle in the air can help calculate the temperature and humidity.

### About the Dataset ###
The dataset was retrieved from University of California Irvine Machine Learning Repository <https://archive.ics.uci.edu/dataset/360/air+quality> as a CSV file and loaded into a Jupyter Notebook. The following columns are available in the dataset.

0. Date	(DD/MM/YYYY)<br>
1. Time	(HH.MM.SS)<br>
2. True hourly averaged concentration CO in mg/m^3  (reference analyzer)<br>
3. PT08.S1 (tin oxide)  hourly averaged sensor response (nominally  CO targeted)	<br>
4. True hourly averaged overall Non Metanic HydroCarbons concentration in microg/m^3 (reference analyzer)<br>
5. True hourly averaged Benzene concentration  in microg/m^3 (reference analyzer)<br>
6. PT08.S2 (titania) hourly averaged sensor response (nominally NMHC targeted)	<br>
7. True hourly averaged NOx concentration  in ppb (reference analyzer)<br>
8. PT08.S3 (tungsten oxide) hourly averaged sensor response (nominally NOx targeted) <br>
9. True hourly averaged NO2 concentration in microg/m^3 (reference analyzer)	<br>
10. PT08.S4 (tungsten oxide) hourly averaged sensor response (nominally NO2 targeted)	<br>
11. PT08.S5 (indium oxide) hourly averaged sensor response (nominally O3 targeted)<br>
12. Temperature in Â°C	<br>
13. Relative Humidity (%) 	<br>
14. AH Absolute Humidity

### Packages
Packages loaded for these analysis are as follows but not all the packages have been utilized:
- pandas
- numpy
- seaborn
- matplotlib
- scipy
- sklearn
- lazypredict
- pytorch & pytorch lightning for neural network

### Data Exploration
The data was explored for null values, negative values and ignoring the outliers for creating a good machine learning model.

### Machine Learning and Deep Learning

####AirQuality_ML
This is the machine learning file which uses various machine learning algorithms excluding the neural network from Pytorch.

Since there were more than one target column, the main task was to make sure that the target values were scaled seperately so that the model can run with high accuracy. TreeRegressor was used to find the best features through feature importance and the best features were selected for the predictions which slightly improves the accuracy of the tree regressor.

LazyPredict library allows to test various models at the same time which was used to get some of the best predictors.

XG Boost which was among the best models from Lazy Predict was used to test with the model and Grid Search was used to improve the parameters for hyperparameter tuning which improves the accuracy.

Since algorithm like SVR did not work with multiple output in the target. MultiOutputRegressor was used to test with further algorithms such as SVR.

The true and predict target are illustrated in a scatter plot for better visualization.

### AirQuality_pytorch
This file is related to deep learning neural network for the same dataset.

The target scaling process have not been repeated in the deep learning task and since it is a multiple output regression further work can be performed to improve the accuracy of the neural network which will be further refined.

### Reflection

This was a really interesting task where there were multiple output that was supposed to be in the target column which made the task more interesting and I learned new tools such as multioutputregressor however, more work needs to be performed in the deep learnig project.