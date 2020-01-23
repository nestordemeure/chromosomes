from fastai.vision import *
from fastai.metrics import error_rate
from tools.temperature_scaling import ModelWithTemperature

#------------------------------------------------------------------------------
# DATA IMPORTATION

# path to data
path = pathlib.PosixPath("../data/png_unscaled_data")

# data importation parameters
valid_pct=0.2 # validation set percentage
size = 224 # picture size
seed = 42 # fix seed for reproducible data split
bs = 64 # batch size

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
data = ImageDataBunch.create_from_ll(src, ds_tfms=tfms, size=size, num_workers=8, bs=bs, resize_method=ResizeMethod.SQUISH, padding_mode='zeros').normalize()

#------------------------------------------------------------------------------
# MODEL

# define model
learn = cnn_learner(data, models.resnet34, pretrained=False, metrics=accuracy, loss_func=LabelSmoothingCrossEntropy())
learn.load("smooth_res34")

# uncalibrated model for comparison
#learn.model = torch.nn.Sequential(ModelWithTemperature(learn.model), torch.nn.Softmax())
#learn.save("uncalibrated_smooth_res34")

#------------------------------------------------------------------------------
# CALIBRATION

# Create a DataLoader from the SAME VALIDATION SET used to train orig_model
# as the process cannot improve the accuracy, it does not matter if it is the validation set
testset_loader = learn.data.dl('test')

# rescale model to calibrate output probabilities
scaled_model = ModelWithTemperature(learn.model)
scaled_model.set_temperature(testset_loader)

# adds sofmax so that probabilities are outputed instead of raw values
scaled_model = torch.nn.Sequential(scaled_model, torch.nn.Softmax())

learn.model = scaled_model
#learn.save("calibrated_smooth_res34")
