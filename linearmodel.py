import numpy as np

def linearmodel(w,xTe):
# INPUT:
# w weight vector (default w=0)
# xTe dxn matrix (each column is an input vector)
#
# OUTPUTS:
#
# preds predictions

    preds = float(np.asarray(w.T) * np.asmatrix(xTe).T)

    # for w_i, x_ij in list(zip(w, xTe)):
    #     preds += w_i * x_ij

    return preds
