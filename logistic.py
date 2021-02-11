import math
import numpy as np

'''

    INPUT:
    xTr dxn matrix (each column is an input vector)
    yTr 1xn matrix (each entry is a label)
    w weight vector (default w=0)

    OUTPUTS:

    loss = the total loss obtained with w on xTr and yTr
    gradient = the gradient at w

    [d,n]=size(xTr);
'''


def logistic(w, xTr, yTr):
    exp_factor = np.exp(np.sum(-yTr.T * xTr.T * w.T, axis=1))
    grad_num = -yTr * exp_factor
    grad_denom = 1 + exp_factor
    gradient = np.sum((grad_num / grad_denom) * xTr, axis=1).reshape((xTr.shape[0], 1))
    loss = np.sum(np.log(1 + exp_factor))
    return loss, gradient
