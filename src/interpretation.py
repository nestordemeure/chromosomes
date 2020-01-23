from fastai.vision import *
from fastai.metrics import error_rate

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

#------------------------------------------------------------------------------
# INTERPRETATION

# accuracy of the model
#learn.validate()[1].item()

# get various informations on the network
interpreter = learn.interpret()

# confusion matrix of the model
interpreter.plot_confusion_matrix(normalize=True, figsize=(12,12), dpi=60)

# function inspired by plot_top_loss
# plots k random elements from the validation set
def _plot_batch(self, k, figsize=(12,12), heatmap:bool=False, heatmap_thresh:int=16, 
                alpha:float=0.6, cmap:str="magma", show_text:bool=True,
                return_fig:bool=None)->Optional[plt.Figure]:
    "Show images in `top_losses` along with their prediction, actual, loss, and probability of actual class."
    if heatmap is None: heatmap = _test_cnn(self.learn.model)
    size_validDataset = len(self.data.dl(self.ds_type))
    tl_idx = torch.LongTensor(k).random_(0, size_validDataset-1)
    classes = self.data.classes
    cols = math.ceil(math.sqrt(k))
    rows = math.ceil(k/cols)
    fig,axes = plt.subplots(rows, cols, figsize=figsize)
    for i,idx in enumerate(tl_idx):
        im,cl = self.data.dl(self.ds_type).dataset[idx]
        cl = int(cl)
        title = f'predicted:{classes[self.pred_class[idx]]}\nactual:{classes[cl]}\nconfidence:{self.preds[idx][self.pred_class[idx]]:.2f}' if show_text else None
        im.show(ax=axes.flat[i], title=title)
        if heatmap:
            mult = self.GradCAM(idx,self.ds_type,heatmap_thresh,image=False)
            if mult is not None:
                sz = list(im.shape[-2:])
                axes.flat[i].imshow(mult, alpha=alpha, extent=(0,*sz[::-1],0), interpolation='bilinear', cmap=cmap)
    if ifnone(return_fig, defaults.return_fig): return fig
# add methods to the class
ClassificationInterpretation.plot_batch = _plot_batch

# some random pictures and their interpretation
interpreter.plot_batch(6, heatmap=True, alpha=0.4)

# most confident misclassifications (with overlaying heatmap)
# predicted class / actual class / loss / probability of actual class
interpreter.plot_top_losses(6, heatmap=True)