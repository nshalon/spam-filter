from numpy import maximum
import numpy as np
from checkgradHingeAndRidge import checkgradHingeAndRidge


def hinge(w,xTr,yTr,lambdaa):
    #
    #
    # INPUT:
    # xTr dxn matrix (each column is an input vector)
    # yTr 1xn matrix (each entry is a label)
    # lambda: regularization constant
    # w weight vector (default w=0)
    #
    # OUTPUTS:
    #
    # loss = the total loss obtained with w on xTr and yTr
    # gradient = the gradient at w

    pred_z = np.sum(xTr.T * w.T, axis=1) * yTr
    gradient = np.sum((xTr * -yTr) * (pred_z < 1), axis=1).reshape((xTr.shape[0], 1))
    gradient += (2 * lambdaa * w)

    loss = np.sum(np.maximum(1 - pred_z, np.zeros(xTr.shape[1]))) + lambdaa * (np.linalg.norm(w) ** 2)

    # pred_z = np.multiply(yTr, (np.asmatrix(w.T) * xTr)) # 1 x n
    # emp_loss_nonzerod = 1 - pred_z
    # emp_loss = np.maximum(emp_loss_nonzerod, np.zeros(emp_loss_nonzerod.shape[1]))
    # loss = np.sum(emp_loss, axis=1) + lambdaa * (float(np.linalg.norm(w)) ** 2)
    #
    # gradient_over_samples = -np.multiply(yTr.T, xTr.T) # n x d
    # gradient_over_samples[np.argwhere(pred_z >= 1), :] = 0
    # gradient = np.sum(gradient_over_samples, axis=0) + (2 * lambdaa * w.T)
    #gradient = (gradient.T / float(np.linalg.norm(gradient)))

    return loss, gradient
