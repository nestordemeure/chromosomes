import pickle  # saving and loading data
import cv2  # image manipulation (python opencv)
import glob  # path manipulation
#import numpy as np  # array manipulation
import random  # for sampling
import os  # folder creation
from sklearn.decomposition import IncrementalPCA  # pca
import matplotlib.pyplot as plot  # for display

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
        pictures = random.sample(pictures, sample_size)
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
training_proportion = 0.8
sample_size = 1865
composant_number = 80

#preprocess_data(input_folder)
#sample_size = split_data(input_folder + "preprocessed/", training_proportion)

#pca = train_pca(input_folder + "preprocessed/" + "training/", sample_size)
#pca = load_data(input_folder + "preprocessed/" + "training/" + "pca_{n}.pca".format(n=sample_size))
#variance_explained_by(pca, composant_number)
#compress_data(input_folder + "preprocessed/" + "training/", pca, composant_number)

#dataset = fuse_dataset(input_folder + "preprocessed/" + "training/" + "compressed/")
dataset = load_data(input_folder + "preprocessed/" + "training/" + "compressed/" + "dataset.pfull")
#displays_dataset(dataset)

#--------------------------------------------------------------------------------------------------

def get_unlabeled_training_sample(input_folder, sample_size):
    """returns an iterator with sample_size element per category in a label*pictures list"""
    for path in glob.glob(input_folder + "*.p"):
        label, pictures = load_data(path)
        sample = random.sample(pictures, sample_size)
        yield from sample
    print("Sampling is finished.", flush=True)

#preprocess_data(input_folder, save_folder)

#sample = list(get_unlabeled_training_sample(save_folder, sample_size))
#save_data(sample, "./sample.p")
#sample = load_data("./sample.p")

# see also Incremental PCA to work on the full dataset while not burning the memory
# http://scikit-learn.org/stable/auto_examples/decomposition/plot_incremental_pca.html#sphx-glr-auto-examples-decomposition-plot-incremental-pca-py
#pca = PCA(n_components=100)
#pca.fit(sample)
#save_data(pca, "./pca.p")
#pca = load_data("./pca.p")

def compress_data(input_folder, output_folder, pca, composant_number):
    variances = pca.explained_variance_ratio_
    variance_represented = sum(variances[0:composant_number])
    print("You can represent {t}% of the original variance using {n} composants.".format(n=composant_number, t=100*variance_represented))
    for path in glob.glob(input_folder + "*.p"):
        label, pictures = load_data(path)
        new_pictures = list(map(lambda tab: tab[0:composant_number], pca.transform(pictures)))
        new_path = path.replace(input_folder, output_folder)
        save_data((label, new_pictures), new_path)
    print("Data has been reduced to {n} composants using the given pca.".format(n=composant_number), flush=True)

#compress_data(save_folder, pca_folder, pca, 80)