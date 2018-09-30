import pickle  # saving and loading data
#import numpy as np  # array manipulation
import random  # for sampling
from sklearn.neighbors import KNeighborsClassifier

#--------------------------------------------------------------------------------------------------
# SAVING

def save_data(data, path):
    """save the data as a pickled file"""
    with open(path, 'wb') as file:
        pickle.dump(data, file)
        print("Data is saved.", flush=True)

def load_data(path):
    """load the data from the disk"""
    with open(path, 'rb') as file:
        data = pickle.load(file)
        print("Data is loaded.", flush=True)
        return data

#--------------------------------------------------------------------------------------------------
# processing

def extract_trainset(dataset, sample_size):
    """converts the dataset into two lists"""
    result_labels = []
    result_points = []
    for(label, points) in dataset:
        points = random.sample(points, sample_size)
        result_points.extend(points)
        result_labels.extend(label*len(points))
    print("the data is now two lists of lenghts {l} and {p}.".format(l=len(result_labels), p=len(result_points)), flush=True)
    return result_labels, result_points

#--------------------------------------------------------------------------------------------------
# KNN



#--------------------------------------------------------------------------------------------------
# TEST

input_folder = "./png_scaled_contrast_data/" + "preprocessed/" + "training/" + "compressed/"
sample_size = 1865  # sample size to train KNN
neighbours_number = 6
axe_number = 45

dataset = load_data(input_folder + "dataset.pfull")
labels, points = extract_trainset(dataset, sample_size)

knn = KNeighborsClassifier(n_neighbors=neighbours_number)
knn.fit(points, labels)