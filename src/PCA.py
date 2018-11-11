import pickle  # saving and loading data
import cv2  # image manipulation (python opencv)
import glob  # path manipulation
#import numpy as np  # array manipulation
import random  # for sampling
import os  # folder creation
from sklearn.decomposition import IncrementalPCA  # pca
import matplotlib.pyplot as plot  # for display
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
# PREPROCESSING

def preprocess_data(input_folder):
    """for each category, load all the pictures and save them as a single file in the input folder"""
    os.makedirs(input_folder + "preprocessed/")
    for folder_path in glob.glob(input_folder + "*"):
        paths = glob.glob(folder_path + "/*.png")
        if len(paths) != 0:
            print("Processing folder " + folder_path, flush=True)
            # get the data
            pictures = [cv2.imread(path, cv2.IMREAD_GRAYSCALE).flatten() for path in paths]
            label = folder_path.replace(input_folder, "")
            # save the data
            save_path = input_folder + "preprocessed/" + label + ".p"
            save_data((label, pictures), save_path)
    print("Preprocessing is finished.", flush=True)

def split_data(input_folder, training_proportion):
    """split dataset into training and testing sets
    returns the size of the smallest training set"""
    os.makedirs(input_folder + "training/")
    os.makedirs(input_folder + "testing/")
    minimum_training_size = float('inf')
    for path in glob.glob(input_folder + "*.p"):
        # get the data ready
        label, pictures = load_data(path)
        random.shuffle(pictures)
        # saves the data
        testing_path = input_folder + "testing/" + label + ".p"
        training_path = input_folder + "training/" + label + ".p"
        training_size = int(training_proportion*len(pictures))
        minimum_training_size = min(minimum_training_size, training_size)
        save_data((label, pictures[training_size:]), testing_path)
        save_data((label, pictures[:training_size]), training_path)
        print("Processed label {l}, training size is {n}".format(l=label, n=training_size), flush=True)
    print("Splitting is finished, the smallest training set is of size {n}.".format(n=minimum_training_size), flush=True)
    return minimum_training_size

def fuse_dataset(input_folder):
    """fuse all the .p files from a folder in a single .pfull file"""
    result = []
    for path in glob.glob(input_folder + "*.p"):
        data = load_data(path)
        result.append(data)
    save_data(result, input_folder + "dataset.pfull")
    print("The dataset has been fused.", flush=True)
    return result

#--------------------------------------------------------------------------------------------------
# PCA

def train_pca(input_folder, sample_size):
    """trains a pca using a random sample of the given size from each label
    the size of the pca is restricted to 100 composants"""
    pca = IncrementalPCA(n_components=100, batch_size=sample_size)
    for path in glob.glob(input_folder + "*.p"):
        label, pictures = load_data(path)
        pictures = random.sample(list(pictures), sample_size)
        pca.partial_fit(pictures)
        print("The PCA has been trained on {l}.".format(l=label), flush=True)
    save_data(pca, input_folder + "pca_{n}.pca".format(n=sample_size))
    print("Training finished.", flush=True)
    return pca

def variance_explained_by(pca, composant_number):
    """returns the fraction of original variance that can be explained by keeping the composant_number first composants"""
    explained_var = sum(pca.explained_variance_ratio_[:composant_number])
    print("{n} composants would preserve {v:2f}% of the original variance.".format(n=composant_number, v=100*explained_var))
    return explained_var

def compress_data(input_folder, pca, composant_number):
    os.makedirs(input_folder + "compressed/")
    for path in glob.glob(input_folder + "*.p"):
        label, pictures = load_data(path)
        pictures = pca.transform(pictures)
        pictures = [pict[:composant_number] for pict in pictures]
        outpath = input_folder + "compressed/" + label + ".p"
        save_data((label, pictures), outpath)
        print("{l} has been compressed.".format(l=label), flush=True)
    print("Compression finished.", flush=True)

#--------------------------------------------------------------------------------------------------
# SCALER

from sklearn.preprocessing import StandardScaler

def train_scaler(input_folder, sample_size):
    scaler = StandardScaler(with_std=False)
    for path in glob.glob(input_folder + "*.p"):
        label, pictures = load_data(path)
        pictures = random.sample(pictures, sample_size)
        scaler.partial_fit(pictures)
        print("The Scaler has been trained on {l}.".format(l=label), flush=True)
    save_data(scaler, input_folder + "scaler_{n}.scaler".format(n=sample_size))
    print("Training finished.", flush=True)
    return scaler

def scale_data(input_folder, scaler):
    os.makedirs(input_folder + "scaled/")
    for path in glob.glob(input_folder + "*.p"):
        label, pictures = load_data(path)
        pictures = scaler.transform(pictures)
        outpath = input_folder + "scaled/" + label + ".p"
        save_data((label, pictures), outpath)
        print("{l} has been scaled.".format(l=label), flush=True)
    print("Scaling finished.", flush=True)

#--------------------------------------------------------------------------------------------------
# PLOT

def displays_dataset(dataset):
    """displays the dataset as a 2D scatterplot using the two main composants"""
    for (label, points) in dataset:
        x = [point[0] for point in points]
        y = [point[1] for point in points]
        plot.scatter(x, y, alpha=0.5, label=label)
    plot.legend(loc='upper right')
    plot.show()

#--------------------------------------------------------------------------------------------------
# TEST

input_folder = "./png_scaled_contrast_data/"
training_proportion = 0.8  # fraction of the dataset that will be used for training
sample_size = 1865  # sample size to train PCA and knn
composant_number = 80  # number of composants kept after compression

#preprocess_data(input_folder)
input_folder += "preprocessed/"
#sample_size = split_data(input_folder, training_proportion)
training_folder = input_folder + "training/"
testing_folder = input_folder + "testing/"

# useless does not improve dataset
#scaler = train_scaler(training_folder, sample_size)
#scale_data(training_folder, scaler)
#scale_data(testing_folder, scaler)
#training_folder += "scaled/"
#testing_folder += "scaled/"
#del scaler

#pca = train_pca(training_folder, sample_size)
#pca = load_data(training_folder + "pca_{n}.pca".format(n=sample_size))
#variance_explained_by(pca, composant_number)
#compress_data(training_folder, pca, composant_number)
#compress_data(testing_folder, pca, composant_number)
training_folder += "compressed/"
testing_folder += "compressed/"
#del pca

dataset = fuse_dataset(training_folder)
#dataset = load_data(training_folder + "dataset.pfull")
#dataset = fuse_dataset(testing_folder)
#dataset = load_data(testing_folder + "dataset.pfull")

#displays_dataset(dataset)