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
    eps = 2.2204e-14  # minimum step size for gradient descent
    t = 0
    loss, gradient = func(w0)
    prev_gradient = gradient
    w = w0[:]
    while t < maxiter and float(np.linalg.norm(gradient)) > tolerance:
        t += 1
        loss, gradient = func(w)
        if np.linalg.norm(gradient) <= np.linalg.norm(prev_gradient):
            stepsize *= 2.0
        else:
            stepsize *= 0.25
        # if t % 1e2 == 0:
            # print("trial %s loss: %s" % (t, loss))
            # print('Gradient Norm: {}'.format(np.linalg.norm(gradient)))
        w = w - stepsize * gradient
        prev_gradient = gradient

    return w
