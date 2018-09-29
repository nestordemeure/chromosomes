# Chromotypage

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



