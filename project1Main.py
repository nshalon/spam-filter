from spamfilter import spamfilter
from trainspamfilter import trainspamfilter
from valsplit import valsplit

from scipy import io
import numpy as np

# load the data:
data = io.loadmat('data/data_train_default.mat')
X = data['X']
Y = data['Y']

# split the data:
xTr, xTv, yTr, yTv = valsplit(X, Y)

xTr_normed = xTr / xTr.sum(axis=0)
xTv_normed = xTv / xTv.sum(axis=0)

# train spam filter with settings and parameters in trainspamfilter.py
w_trained = trainspamfilter(xTr_normed, yTr)

# evaluate spam filter on test set using default threshold
spamfilter(xTv_normed, yTv, w_trained)
