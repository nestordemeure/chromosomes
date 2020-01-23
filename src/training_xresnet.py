from fastai.vision import *
from fastai.metrics import error_rate
from ranger import *

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

# remove expansion input (workaround to deal with current xresnet implem)
def noexpand(model, expansion=None, **kwargs):
    return model(**kwargs)
myxresnet34 = partial(noexpand, models.xresnet34)

# define model
learn = cnn_learner(data, models.resnet34, pretrained=False, metrics=accuracy, loss_func=LabelSmoothingCrossEntropy())
#learn.load("xres")

# finds a good learning rate
learn.lr_find()
learn.recorder.plot()

learn.fit_one_cycle(5, max_lr=1e-2)
learn.recorder.plot_losses()

learn.fit_one_cycle(5, max_lr=2e-3)
learn.recorder.plot_losses()

# 0.?
# ?? min per step
#learn.save("smooth")

# 0.90
# 33 min per step
#learn.save("xres")

#------------------------------------------------------------------------------

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
learn = cnn_learner(data, myxresnet34, pretrained=False, metrics=accuracy, 
                    loss_func=LabelSmoothingCrossEntropy())
learn.fit_one_cycle(20, max_lr=1e-2)
learn.recorder.plot_losses()

# 0.922
# 28 min per step
opt_func = partial(Ranger, betas=(0.95,0.99), eps=1e-6)
learn = cnn_learner(data, models.resnet34, pretrained=False, metrics=accuracy, 
                    loss_func=LabelSmoothingCrossEntropy(), opt_func=opt_func)
learn.fit_one_cycle(20, max_lr=1e-2)
learn.recorder.plot_losses()

learn.save("smooth_res34")