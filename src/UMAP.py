import pickle  # saving and loading data
import random  # for sampling
import umap
import matplotlib.pyplot as plot
import seaborn as sns

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

#--------------------------------------------------------------------------------------------------
# TEST

input_folder = "./png_scaled_contrast_data/" + "preprocessed/"
train_sample_size = 1865
test_sample_size = 467

trainset = load_data(input_folder + "training/" + "compressed/" + "dataset.pfull")
trainlabels, trainpoints = extract_trainset(trainset, train_sample_size)
trainlabels = [int(x)-1 for x in trainlabels]
del trainset

#reducer = umap.UMAP(n_neighbors=35, min_dist=0.5, metric='minkowski')
reducer = umap.UMAP(n_neighbors=30, min_dist=0.8, metric='cosine')
embedding = reducer.fit_transform(trainpoints, y=trainlabels)

plot.scatter(embedding[:, 0], embedding[:, 1], alpha=0.5, c=[sns.color_palette(n_colors=24)[x] for x in trainlabels])
plot.gca().set_aspect('equal', 'datalim')
plot.show()

testset = load_data(input_folder + "testing/" + "compressed/" + "dataset.pfull")
testlabels, testpoints = extract_trainset(testset, test_sample_size)
testlabels = [int(x)-1 for x in testlabels]
del testset

testembedding = reducer.transform(testpoints)

plot.scatter(testembedding[:, 0], testembedding[:, 1], alpha=0.5, c=[sns.color_palette(n_colors=24)[x] for x in testlabels])
plot.gca().set_aspect('equal', 'datalim')
plot.show()

#--------------------------------------------------------------------------------------------------
# KNN

from sklearn.neighbors import KNeighborsClassifier

neighbours_number = 70

knn = KNeighborsClassifier(n_neighbors=neighbours_number)
knn.fit(embedding, trainlabels)
score = knn.score(testembedding, testlabels)
print("The score for k={k} is {s}".format(k=neighbours_number, s=score))
