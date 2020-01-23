from fastai.vision import *

# path to data
path = pathlib.PosixPath("../data/png_unscaled_data")

# validation set percentage
valid_pct=0.2
# picture size
size = 224
# fix seed for reproducible data split
seed = 42

# defines transformations that will be used to augment the training set :
# lighting is constant so not modified
# vertical flips are not meaningful as the pictures have been oriented
# horizontal flips seem to be meaningful (despite the possibility of creating mirror chromosomes)
tfms = get_transforms(max_lighting=None, do_flip=True, flip_vert=False)

# imports a list of pictures
# (takes the negativ of the picture so that black is the background color)
il = ImageList.from_folder(path, after_open=PIL.ImageOps.invert)
# splits into train/test and adds labels
src = il.split_by_rand_pct(valid_pct, seed=seed).label_from_folder()
# builds a databunch
data = ImageDataBunch.create_from_ll(src, ds_tfms=tfms, size=size, num_workers=8, resize_method=ResizeMethod.SQUISH, padding_mode='zeros').normalize()

#------------------------------------------------------------------------------
# BAYESIAN LOSS

# stores a confidence values for each of the output class
# it reflects how much we trust the labels for that class (and, indirectly, ourselves)
class ConfidenceLayer(Module):
    def __init__(self, nbClass):
        super(ConfidenceLayer, self).__init__()
        # confidence initialized to 5 which is equivalent to 99% after a sigmoid
        confidence = torch.Tensor(1, nbClass)
        confidence.fill_(5.)
        self.confidence = nn.Parameter(confidence.cuda()) # .cuda not needed on collaboratoty :o
    # adds a field confidence to the tensor
    def forward(self, input):
        input.confidence = self.confidence
        return input

# put on top of a bayesian network to get a standard network
class ProbabilityLayer(Module):
    # adds a field confidence to the tensor
    def forward(self, input):
        # insures that all values are in [0,1] and that they sum to 1
        probabilities = F.softmax(input, dim=-1)
        # insures that all values are in [0,1]
        confidence = torch.sigmoid(input.confidence)
        # output probability
        probability = probabilities*confidence
        return probability

# loss that is maximized by giving the proper probability to ones guess
# relies on the fast that output was produced by a ConfidenceLayer
# TODO: this could be made much more numerically stable using LogSumExp
def correctProbability_loss(output, target):
    # insures that all values are in [0,1] and that they sum to 1
    probabilities = F.softmax(output, dim=-1)
    # insures that all values are in [0,1]
    confidence = torch.sigmoid(output.confidence)
    # probabilities*confidence + (1 - probabilities)*(1 - confidence)
    correctedProbability = torch.log(2*probabilities*confidence - (probabilities + confidence) + 1) 
    # -sum(correctedProbability[target])
    loss = F.nll_loss(correctedProbability, target, reduction='sum')
    # minimize -sum(log(proba)) <=> maximise product(proba)
    return loss

# takes a model and modifies it to make it bayesian
# we recommand using a pretrained model
def bayesian_cnn_learner(data=None, model=None, pretrained=False, metrics=accuracy, **kwargs):
    learner = cnn_learner(data, model, pretrained=pretrained, metrics=metrics, **kwargs)
    # the model is frozen as we do not need to train the inner classes
    if len(learner.layer_groups) > 1: learner.freeze()
    else: print("WARNING: Unable to freeze layers which might mean that the model does not have a head (which would cause an error later during the initialization).")
    # a final layer is added to store the confidence information
    nbClass = learner.data.c
    learner.model = nn.Sequential(learner.model, ConfidenceLayer(nbClass))
    # the loss is replaced by a loss that takes confidence information into account
    learner.old_loss = learner.loss_func
    learner.loss_func = correctProbability_loss
    return learner

#------------------------------------------------------------------------------
# TRAINING

# define model
#learn = cnn_learner(data, models.resnet34, pretrained=False, metrics=accuracy, loss_func=LabelSmoothingCrossEntropy())
learn = bayesian_cnn_learner(data, models.resnet34, pretrained=False, 
                             metrics=[accuracy], wd=0)
#learn.load("bayesian")

# finds a good learning rate
learn.lr_find()
learn.recorder.plot()

# 0.864
# 20 min par iter
learn.fit_one_cycle(9, max_lr=1e-2/5)
learn.recorder.plot_losses()

learn.validate(metrics=[accuracy])

#learn.save("bayesian")

#------------------------------------------------------------------------------
# TESTS

# 0.864
# 20 min par iter
# learn = bayesian_cnn_learner(data, models.resnet34, pretrained=False, metrics=[accuracy,correctAccuracy])
# learn.fit_one_cycle(20, max_lr=1e-2)
# learn.recorder.plot_losses()

# 0.9218
# 28 min per step
#learn = cnn_learner(data, models.resnet34, pretrained=False, metrics=accuracy)
#learn.fit_one_cycle(20, max_lr=1e-2)
#learn.recorder.plot_losses()

# 0.923
# 28 min per step
#learn = cnn_learner(data, models.resnet34, pretrained=False, metrics=accuracy, 
#                    loss_func=LabelSmoothingCrossEntropy())
#learn.fit_one_cycle(20, max_lr=1e-2)
#learn.recorder.plot_losses()

# 0.923
# 33 min per step
#learn = cnn_learner(data, myxresnet34, pretrained=False, metrics=accuracy)
#learn.fit_one_cycle(20, max_lr=1e-2)
#learn.recorder.plot_losses()

# 0.924
# 32 min per step
#learn = cnn_learner(data, myxresnet34, pretrained=False, metrics=accuracy, 
#                    loss_func=LabelSmoothingCrossEntropy())
#learn.fit_one_cycle(20, max_lr=1e-2)
#learn.recorder.plot_losses()
