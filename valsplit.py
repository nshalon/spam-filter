import math
import numpy as np


def valsplit(X,Y):
    _, n = np.shape(X)
    part = int(math.ceil(n*0.8))
    xTr = X[:, 0:part].toarray()
    xTv = X[:, part:n].toarray()
    yTr = Y[:, 0:part]
    yTv = Y[:, part:n]
    return xTr, xTv, yTr, yTv
