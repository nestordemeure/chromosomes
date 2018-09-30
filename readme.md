# Chromotypage

# Pipeline

## Classification

- adjust histogram for better contrast
- rescale pictures to identical size
- run PCA to reduce dimensions
- classify with KNN

## Training

- adjust histogram and rescale all pictures
- split picture into training dataset (80%) and testing dataset (20%)
- train PCA and save reduced pictures
- train KNN and crossvalidate to identify optimal K and composant number

# Task

- Given a picture, returns one of the 24 possible labels.
- Given a picture, return one of the 24 possible labels or one of the extra (q or phi).

Validation of the algorithm will be done on 10% of the dataset (using data that was not given for training).

*Warning* : The data are not balanced between labels and, in particular, there are less data for the 23/24 labels and a lot less data for the anomalies.

# Algorithm

Something similar to [how-to-get-97-percent-on-MNIST-with-KNN](https://steven.codes/blog/ml/how-to-get-97-percent-on-MNIST-with-KNN/) :
- take the png picture
- center them in a rectangle of given size (use center of mass or just center of picture ?)
- adjust the contrast
- perform a PCA to reduce the dimensionality (and vizualize the result)
- use the k nearest neigbours algorithm to classify the data

We can optimize the number of dimension kep and the number of neigbours.



