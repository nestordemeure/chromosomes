import pickle  # saving and loading data
#import numpy as np  # array manipulation
import random  # for sampling
from sklearn.neighbors import KNeighborsClassifier
import os  # folder creation

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

def extract_trainset(dataset, sample_size=None):
    """converts the dataset into two lists"""
    result_labels = []
    result_points = []
    for(label, points) in dataset:
        if not (sample_size is None):
            points = random.sample(points, sample_size)
        labels = [label]*len(points)
        result_points.extend(points)
        result_labels.extend(labels)
    print("the data is now two lists of lenght {l}.".format(l=len(result_points)), flush=True)
    return result_labels, result_points

def crop_axis(points, axis_number):
    """keeps only the axis_number most important axis"""
    return [point[:axis_number] for point in points]

def max_sample_size(dataset):
    """extract the length of the smallest label"""
    result = float('inf')
    for label, points in dataset:
        result = min(result, len(points))
    print("The maximum possible sample size for the dataset is {n}".format(n=result))
    return result

#--------------------------------------------------------------------------------------------------
# KNN

def score_model(input_folder, neighbours_number, axe_number, trainlabels, trainpoints, testlabels, testpoints):
    knn = KNeighborsClassifier(n_neighbors=neighbours_number)
    knn.fit(trainpoints, trainlabels)
    save_data(knn, input_folder + "{k}nn_{n}axes.model".format(k=neighbours_number, n=axe_number))
    score = knn.score(testpoints, testlabels)
    print("The score for k={k} and {n} axis is {s}".format(k=neighbours_number, n=axe_number, s=score))
    return score

def cross_validate(input_folder, max_neighbours_number, max_axe_number, trainlabels, trainpoints, testlabels, testpoints):
    input_folder = input_folder + "model/"  # folder where we will store all the intermediate models
    os.makedirs(input_folder)
    score = 0
    bestk = 0
    besta = 0
    scores = []
    ks = []
    axes = []
    for a in reversed(range(1,max_axe_number, 5)):
        trainpoints = crop_axis(trainpoints, a)
        testpoints = crop_axis(testpoints, a)
        for k in range(1, max_neighbours_number, 2):
            new_score = score_model(input_folder, k, a, trainlabels, trainpoints, testlabels, testpoints)
            scores.append(new_score)
            ks.append(k)
            axes.append(a)
            if new_score > score:
                score = new_score
                bestk = k
                besta = a
    save_data((scores, ks, axes), input_folder + "ska.scores")
    print("The best score was {s} which is reached at k={k} and {a} axis.".format(s=score, k=bestk, a=besta))
    return scores, ks, axes

#--------------------------------------------------------------------------------------------------
# TEST

input_folder = "./png_scaled_contrast_data/" + "preprocessed/"
train_sample_size = 1865
test_sample_size = 467

trainset = load_data(input_folder + "training/" + "compressed/" + "dataset.pfull")
#train_sample_size = max_sample_size(trainset)
trainlabels, trainpoints = extract_trainset(trainset, train_sample_size)
del trainset

testset = load_data(input_folder + "testing/" + "compressed/" + "dataset.pfull")
#test_sample_size = max_sample_size(testset)
testlabels, testpoints = extract_trainset(testset, test_sample_size)
del testset

#max_neighbours_number = 50
#max_axe_number = 80
#scores, neigbours, axes = cross_validate(input_folder, max_neighbours_number, max_axe_number,
#                                         trainlabels, trainpoints, testlabels, testpoints)
#scores, neigbours, axes = load_data(input_folder + "model/" + "ska.scores")

#--------------------------------------------------------------------------------------------------
# grid search

from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix


class Crop():
    axis_number = 1
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return crop_axis(X, self.axis_number)
    def set_params(self, axis_number):
        self.axis_number = axis_number
        return self


parameters = dict(knn__n_neighbors=range(1,50,2), crop__axis_number=range(2,100))
classifier = Pipeline([('crop',Crop()), ('knn',KNeighborsClassifier())])
gridSearch = GridSearchCV(classifier, parameters, cv=3, scoring='precision_weighted', verbose=2)
gridSearch.fit(trainpoints, trainlabels)

bestparam = gridSearch.best_params_
bestscore = gridSearch.best_score_
score = gridSearch.score(testpoints, testlabels)
print("The best score is {bs} ({s} on test set) which was reached at {p}.".format(bs=bestscore, s=score, p=bestparam))
#The best score is 0.5492849131003329 (0.5654278291654744 on test set) which was reached at {'crop__axis_number': 80, 'knn__n_neighbors': 35}.