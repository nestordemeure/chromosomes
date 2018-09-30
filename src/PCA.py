import cv2  # image manipulation (python opencv)
import glob  # path manipulation
import numpy as np  # array manipulation
from sklearn.decomposition import PCA  # pca
import pickle  # saving and loading data
import random  # for sampling
import matplotlib.pyplot as plot  # for display


input_folder = "./png_scaled_contrast_data/"
save_folder = "./pickle/"
pca_folder = "./pca80/"
sample_size = 100

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

def preprocess_data(input_folder, save_folder):
    """for each category, load all the pictures and save them as a single file"""
    for folder_path in glob.glob(input_folder + "*"):
        print("Processing folder " + folder_path, flush=True)
        # get the data
        label = folder_path.replace(input_folder, "")
        paths = glob.glob(folder_path + "/*.png")
        pictures = [cv2.imread(path, cv2.IMREAD_GRAYSCALE).flatten() for path in paths]
        # save the data
        save_path = folder_path.replace(input_folder, save_folder) + ".p"
        save_data((label, pictures), save_path)
    print("Preprocessing is finished.", flush=True)

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
sample = load_data("./sample.p")

# see also Incremental PCA to work on the full dataset while not burning the memory
# http://scikit-learn.org/stable/auto_examples/decomposition/plot_incremental_pca.html#sphx-glr-auto-examples-decomposition-plot-incremental-pca-py
#pca = PCA(n_components=100)
#pca.fit(sample)
#save_data(pca, "./pca.p")
pca = load_data("./pca.p")

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

compress_data(save_folder, pca_folder, pca, 80)