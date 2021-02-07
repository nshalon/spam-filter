import numpy as np

def grdescent(func, w0, stepsize, maxiter, tolerance=1e-02):
# INPUT:
# func function to minimize
# w_trained = initial weight vector
# stepsize = initial gradient descent stepsize
# tolerance = if norm(gradient)<tolerance, it quits
#
# OUTPUTS:
#
# w = final weight vector
    eps = 2.2204e-14 #minimum step size for gradient descent
    iter = 0
    loss, gradient = func(w0)
    w = w0[:]
    while iter < maxiter and float(np.linalg.norm(gradient)) > tolerance:
        iter += 1
        loss, gradient = func(w)
        if iter % 1e2 == 0:
            print("trial %s loss: %s" % (iter, loss))
        w = w - stepsize * gradient

    return w
