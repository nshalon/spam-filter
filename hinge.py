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
    pred_z = np.multiply(yTr, (np.asmatrix(w.T) * xTr)) # 1 x n
    emp_loss_nonzerod = 1 - pred_z
    emp_loss = np.maximum(emp_loss_nonzerod, 0)
    loss = np.sum(emp_loss, axis=1) + lambdaa * (np.linalg.norm(w) ** 2)

    gradient_over_samples = np.multiply(-1*yTr.T, xTr.T) # n x d
    gradient_over_samples[np.argwhere(pred_z.flatten() > 1), :] = 0
    gradient = np.sum(gradient_over_samples, axis=0) + 2*(lambdaa * w.T)

    return loss, gradient.T
