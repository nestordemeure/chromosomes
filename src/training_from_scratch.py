from fastai.vision import *
from fastai.metrics import error_rate

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

# defines model
learn = cnn_learner(data, models.resnet34, pretrained=False, metrics=accuracy)
#learn.load("scratch")

# finds a good learning rate
learn.lr_find()
learn.recorder.plot()

learn.fit_one_cycle(5, max_lr=1e-2)
learn.recorder.plot_losses()

learn.fit_one_cycle(5, max_lr=1e-3)
learn.recorder.plot_losses()

learn.fit_one_cycle(5, max_lr=1e-4)
learn.recorder.plot_losses()

# 0.91
# 28 min per step
#learn.save("scratch")
