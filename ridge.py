
import numpy as np


def ridge(w, xTr, yTr, lambdaa):
#
# INPUT:
# w weight vector (default w=0)
# xTr:dxn matrix (each column is an input vector)
# yTr:1xn matrix (each entry is a label)
# lambdaa: regression constant
#
# OUTPUTS:
# loss = the total loss obtained with w on xTr and yTr
# gradient = the gradient at w
#
# [d,n]=size(xTr);

    # YOUR CODE HERE
    sum_term = np.sum(xTr.T * w.T, axis=1) - yTr
    gradient = 2 * np.sum(sum_term * xTr, axis=1).reshape((xTr.shape[0], 1))
    gradient += (2 * lambdaa * w)

    loss = np.sum(sum_term ** 2) + (lambdaa * (np.linalg.norm(w) ** 2))
    return loss, gradient
