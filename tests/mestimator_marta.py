from __future__ import division  # take the division operator from future versions

import numpy as np
import toolboxutilities as util
import sys


# -------------------------------------------------------------------
# Phi functions for the robust regressors
# -------------------------------------------------------------------
def scorefunction(u, kind, clipping):
    if kind == 'huber':  # least squares
        score = huber(u, clipping)  # get the estimate
    elif kind == 'squared':
        score = u
    elif kind == 'optimal':
        score = util.scoreoptimal(u, clipping)
    elif kind == 'tau':
        weighttau = util.tauweights(u, 'optimal', clipping)
        score = (
            weighttau * util.scoreoptimal(u, clipping[0]) + util.scoreoptimal(
                u, clipping[1]
            )
        )
    else:  # unknown method
        sys.exit('unknown method %s' % kind)  # die gracefully
    return score  # return the score function that we need


# -------------------------------------------------------------------
# Huber score function
# -------------------------------------------------------------------
def huber(u, clipping):
    u = np.array(u)  # converting to np array
    p = np.zeros(u.shape)  # new array for the output
    u_abs = np.abs(u)
    i = u_abs <= clipping  # logical array
    p[i] = u[i]  # middle part of the function
    i = u_abs > clipping  # outer part of the function
    p[i] = np.sign(u[i]) * clipping
    return p


# -------------------------------------------------------------------
# Weight functions for the IRLS
# -------------------------------------------------------------------
def weights(u, kind, lossfunction, clipping, nmeasurements):
    if kind == 'M':  # if M-estimator
        if lossfunction == 'huber':  # with Huber loss function
            # call the huber score function
            z = scorefunction(u, 'huber', clipping)
            w = np.zeros(u.shape)
            i = np.nonzero(u)
            # definition of the weights for M-estimator
            w[i] = z[i] / (2 * u[i])
        elif lossfunction == 'squared':  # with square function
            # call the ls score function
            z = scorefunction(u, 'squared', clipping)
            w = np.zeros(u.shape)
            i = np.nonzero(u)
            w[i] = z[i] / (2 * u[i])
        elif lossfunction == 'optimal':
            z = scorefunction(u, 'optimal', clipping)
            w = np.zeros(u.shape)
            i = np.nonzero(u)
            w[i] = z[i] / (2 * u[i])
        else:  # unknown loss function
            # die gracefully
            sys.exit('unknown type of loss function %s' % lossfunction)
    elif kind == 'tau':  # if tau estimator
        z = scorefunction(u, 'tau', clipping)
        w = np.zeros(u.shape)
        i = np.nonzero(u)
        # only for the non zero u elements
        w[i] = z[i] / (2 * nmeasurements * u[i])
    else:  # unknown method
        sys.exit('unknown type of weights %s' % kind)  # die gracefully
    return w


# -------------------------------------------------------------------
# Least squares
# -------------------------------------------------------------------
def leastsquares(y, a):
    xhat = np.linalg.lstsq(a, y)[0]  # plain vanilla least squares
    return xhat


def irls(y,
         a,
         lossfunction,
         initialx,
         clipping,
         kind='M',
         maxiter=100,
         tolerance=1e-5
         ):
    """
    Iterative Re-weighted Least Squares algorithm

    Input arguments:
    y: measurements
    a: model matrix
    lossfunction: type of loss function that we want (squared, huber)
    initialx: initial solution
    clipping: clipping parameter for the loss function
    tolerance: if two consecutive iterations give two solutions closer than tolerance, the algorithm stops
    maxiter: maximum number of iterations. If the algorithm reaches this, it stops
    """
    #  Initialization
    m, n = a.shape  # number of measurements and number of unknowns
    y = np.array(y)  # cast to numpy array
    res = y - np.dot(a, initialx)  # initial residuals
    x = initialx  # initialize solution
    xdis = 2 * tolerance  # minimum distance between two consecutive solutions
    k = 0  # iterations counter initialization
    w = 0  # initialization
    steps = np.zeros((maxiter, 1))
    # ------- Iterating -----------------------------------------
    while xdis > tolerance and k < maxiter:
        # while we do not reach any stop condition
        w = util.weights(res, kind, lossfunction, clipping, m)  # getting the
        # weights we need
        sqw = np.sqrt(w)  # to convert it to a matrix multiplication
        aw = a * sqw
        yw = y * sqw  # now these are the LS arguments
        newx = leastsquares(yw, aw)  # solving the weighted LS system
        xdis = np.linalg.norm(x - newx)  # normalized distance
        steps[k] = xdis  # step in this iteration. To check convergence
        res = y - np.dot(a, newx)  # new residual
        x = newx  # update x
        k += 1  # update iter
        # ---------- Wrapping up ----------------------------------------
    return x, k, w, steps


# -------------------------------------------------------------------
# M - estimator
# -------------------------------------------------------------------
def mestimator(y, a, lossfunction, clipping):
    m, n = a.shape
    initialx = np.ones((n, 1))  # initial solution
    xhat, k, w, steps = irls(y, a, lossfunction, initialx, clipping)
    return xhat, steps


def main():
    # usage
    y = np.ones((8, 1))
    a = np.ones((8, 2))
    # for a large clipping parameter, LS and M outputs should be the same
    out, steps = mestimator(y, a, 'huber', 6)
    outls = leastsquares(y,a)
    print out
    print outls

if __name__ == '__main__':
    main()
