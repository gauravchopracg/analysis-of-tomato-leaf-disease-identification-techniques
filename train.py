from fastai.imports import *
from fastai.transforms import *
from fastai.conv_learner import *
from fastai.model import *
from fastai.dataset import *
from fastai.sgdr import *
from fastai.plots import *


PATH = "/home/"
sz = 224
arch=resnet18
bs=64
label_csv = f'train.csv'
n = len(list(open(label_csv))) - 1 # header is not counted (-1)
val_idxs = get_cv_idxs(n) # random 20% data for validation set


tfms = tfms_from_model(arch, sz, aug_tfms=transforms_side_on, max_zoom=1.1)
data = ImageClassifierData.from_csv(PATH, 'images', f'train.csv', 
                                   val_idxs=val_idxs, tfms=tfms, bs=bs)
learn = ConvLearner.pretrained(resnet18, data, precompute=True, opt_fn=optim.Adam, ps=0.5)

learn.fit(1e-3, 4, cycle_len=1, cycle_mult=2)
lrs=np.array([1e-5,1e-4,1e-3])
learn.precompute=False

learn.unfreeze()
lrf=learn.lr_find(lrs/1e3)
learn.sched.plot()
lrs=np.array([1e-5,1e-4,1e-3])
learn.fit(lrs, 4, cycle_len=1, cycle_mult=2)

log_preds, y = learn.TTA()
probs = np.mean(np.exp(log_preds),0)
accuracy_np(probs, y), metrics.log_loss(y, probs)
