"""
===================================================================
UTILITIES
This toolbox contains many useful functions, which are not related
to solving the inverse problem itself.
Marta Martinez-Camara, EPFL
===================================================================
"""

# take the division operator from future versions
from __future__ import division


# -------------------------------------------------------------------
# Getting the source vector
# -------------------------------------------------------------------
def getsource(sourcetype, sourcesize, k=1):
    import numpy as np
    import pickle
    import sys  # to be able to exit
    if sourcetype == 'random':  # Gaussian iid source, mu = 0, sigma = 1
        x = np.random.randn(sourcesize, 1)  # source vector
    elif sourcetype == 'sparse':
        sparsity = k * sourcesize  # 20 % sparsity for the source
        x = np.zeros((sourcesize, 1))  # initialization with a zero source
        p = np.random.permutation(sourcesize)
        nonz = p[0:sparsity]  # getting random indexes for the nonzero values
        # get randomly the value for the non zero elements
        x[nonz] = np.random.randn(sparsity, 1)
    elif sourcetype == 'constant':
        x = np.zeros((sourcesize, 1))  # initialization with a zero source
        x[7:15] = 1  # making a piecewise source

    else:   # unknown type of source
        sys.exit('unknown source type %s' % sourcetype)  # die gracefully
    x = np.asarray(x)
    return x  # what we were asked to deliver


# -------------------------------------------------------------------
# Getting the sensing matrix
# -------------------------------------------------------------------
def getmatrix(sourcesize, matrixtype, measurementsize, conditionnumber=1):
    import numpy as np
    import pickle
    import sys  # to be able to exit
    if matrixtype == 'random':  # Gaussian iid matrix, mu = 0, sigma = 1
        a = np.random.randn(measurementsize, sourcesize)  # sensing matrix
    elif matrixtype == 'illposed':
        # random well conditioned matrix
        a = np.random.randn(measurementsize, sourcesize)
        u, s, v = np.linalg.svd(a)  # get the svd decomposition
        nsv = min(sourcesize, measurementsize)  # number of sv
        # modify the sv to make cond(A) = conditionnumber
        s[np.nonzero(s)] = np.linspace(conditionnumber, 1, nsv)
        sm = np.zeros((measurementsize, sourcesize))
        sm[:sourcesize, :sourcesize] = np.diag(s)
        a = np.dot(u, np.dot(sm, v))  # putting everything together
    else:   # unknown type of matrix
        sys.exit('unknown matrix type %s' % matrixtype)  # die gracefully
    a = np.asarray(a)
    return a  # what we were asked to deliver


# -------------------------------------------------------------------
# Getting the measurements
# -------------------------------------------------------------------
def getmeasurements(a, x, noisetype, var=1, outlierproportion=0):
    import numpy as np
    import pickle
    import sys  # to be able to exit
    import matplotlib.pyplot as plt
    # import statistics as st
    measurementsize = a.shape[0]  # number of measurements
    y = np.dot(a, x)  # noiseless measurements
    if noisetype == 'none':  # noiseless case
        n = np.zeros((measurementsize, 1))  # zero noise
    elif noisetype == 'gaussian':  # gaussian noise
        n = var * np.random.randn(measurementsize, 1)
    elif noisetype == 'outliers':  # gaussian noise
        # additive Gaussian noise
        n = var * np.random.randn(measurementsize, 1)
        p = np.random.permutation(measurementsize)
        # how many measurements are outliers
        noutliers = np.round(outlierproportion * measurementsize)
        outindex = p[0:noutliers]  # getting random indexes for the outliers
        # the outliers have a variance ten times larger than clean data
        n[outindex] = np.var(y) * 10 * np.random.randn(noutliers, 1)

    else:  # unknown type of additive noise
        sys.exit('unknown noise type %s' % noisetype)  # die gracefully
    yn = y + n  # get the measurements
    yn = np.asarray(yn)
    #plt.stem(n, 'b')
    # plt.show(block=True)
    #plt.stem(y, 'kd-')
    #plt.stem(yn, 'rs--')
    # plt.show()  # show figure

    return yn  # what we were asked to deliver


# -------------------------------------------------------------------
# Score functions for the robust regressors
# -------------------------------------------------------------------
def scorefunction(u, kind, clipping):
    import sys  # to tbe able to exit
    if kind == 'huber':  # least squares
        score = huber(u, clipping)  # get the estimate
    elif kind == 'squared':
        score = u
    elif kind == 'optimal':
        score = scoreoptimal(u, clipping)
    elif kind == 'tau':
        # here we compute the score function for the tau.
        # psi_tau = weighttau * psi_1 + psi_2
        weighttau = tauweights(u, 'optimal', clipping)
        # print weighttau

        #weighttau = lp.tau_weights_new(u, clipping)

        score = weighttau * \
            scoreoptimal(u, clipping[0]) + scoreoptimal(u, clipping[1])
    else:  # unknown method
        sys.exit('unknown method %s' % kind)  # die gracefully
    return score  # return the score function that we need


# -------------------------------------------------------------------
# Huber score function
# -------------------------------------------------------------------
def huber(u, clipping):
    import numpy as np
    u = np.array(u)  # converting to np array
    p = np.zeros(u.shape)  # new array for the output
    u_abs = np.abs(u)
    i = u_abs <= clipping  # logical array
    p[i] = u[i]  # middle part of the function
    i = u_abs > clipping  # outer part of the function
    p[i] = np.sign(u[i]) * clipping
    return p


# -------------------------------------------------------------------
# Optimal score function
# -------------------------------------------------------------------
def scoreoptimal(u, clipping):
    import numpy as np
    u = np.array(u)
    p = np.zeros(u.shape)
    uabs = np.abs(u)  # u absolute values
    i = uabs <= 2 * clipping  # central part of teh score function
    p[i] = u[i] / clipping ** 2 / 3.25
    i = np.logical_and(uabs > 2 * clipping, uabs <= 3 * clipping)
    f = lambda z: (-1.944 * z / clipping ** 2 + 1.728 * z ** 3 / clipping ** 4 - 0.312 * z ** 5 / clipping ** 6 +
                   0.016 * z ** 7 / clipping ** 8) / 3.25
    p[i] = f(u[i])
    return p


# -------------------------------------------------------------------
# Rho functions
# -------------------------------------------------------------------
def rhofunction(u, kind, clipping):
    import sys  # to tbe able to exit
    if kind == 'optimal':  # least squares
        r = rhooptimal(u, clipping)  # get the estimate
    else:  # unknown method
        sys.exit('unknown rho function %s' % kind)  # die gracefully
    return r  # return the score function that we need


# -----------------------------------------------
# Optimal loss function (rho)
# -------------------------------------------------------------------
def rhooptimal(u, clipping):
    """
    The Fast-Tau Estimator for Regression, Matias SALIBIAN-BARRERA, Gert WILLEMS, and Ruben ZAMAR
    www.tandfonline.com/doi/pdf/10.1198/106186008X343785

    The equation is found p. 611. To get the exact formula, it is necessary to use 3*c instead of c.
    """

    import numpy as np
    u = np.array(u)
    y = np.abs(u / clipping)
    r = np.ones(u.shape)
    i = y <= 2.  # middle part of the curve
    r[i] = y[i] ** 2 / 2. / 3.25
    i = np.logical_and(y > 2, y <= 3)  # intermediate part of the curve
    f = lambda z: (1.792 - 0.972 * z ** 2 + 0.432 * z **
                   4 - 0.052 * z ** 6 + 0.002 * z ** 8) / 3.25
    r[i] = f(y[i])
    return r


# -------------------------------------------------------------------
# Weight for the score in the tau
# -------------------------------------------------------------------
def tauweights(u, lossfunction, clipping):
    """
    This routine computes the 'weighttau', necessary to build the psi_tau function
    :param u: vector with all arguments we pass to the weights. so we just need to compute to compute this value once
              to find the psi_tau
    :param lossfunction: huber, bisquare, optimal, etc
    :param clipping: the two values of the clipping parameters corresponding to rho_1, rho_2
    :return:
    """

    import numpy as np
    import sys
    if np.sum(scoreoptimal(u, clipping[0]) * u) == 0:
        #  return np.zeros(u.shape)
        return np.ones(u.shape)
    if lossfunction == 'optimal':  # weights for the rho tau.
        w = (np.sum(2. * rhooptimal(u, clipping[1]) - scoreoptimal(u, clipping[1]) * u)
             ) / np.sum(scoreoptimal(u, clipping[0]) * u)
    else:
        sys.exit('unknown type of loss function %s' %
                 lossfunction)  # die gracefully
    return w


# -------------------------------------------------------------------
# Weight functions for the IRLS
# -------------------------------------------------------------------
def weights(u, kind, lossfunction, clipping, nmeasurements):
    import sys  # to be able to exit
    import numpy as np
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
            sys.exit('unknown type of loss function %s' %
                     lossfunction)  # die gracefully
    elif kind == 'tau':  # if tau estimator
        # I called scorefunction to our psi function
        z = scorefunction(u, 'tau', clipping)
        # if r = zero, weights are equal to one
        w = np.ones(u.shape)

        # only for the non zero u elements
        i = np.nonzero(u)
        w[i] = z[i] / (2 * nmeasurements * u[i])
    else:
        # unknown method
        sys.exit('unknown type of weights %s' % kind)  # die gracefully

    return w


# -------------------------------------------------------------------
# M - scale estimator function
# -------------------------------------------------------------------
def mscaleestimator(u, tolerance, b, clipping, kind):
    import numpy as np
    maxiter = 100

    #TODO : changed by Guillaume
    u = np.array(u)

    s = np.median(np.abs(u)) / .6745  # initial MAD estimation of the scale
    # if (s==0):
      # s=1.0
    rho_old = np.mean(rhofunction(u / s, kind, clipping)) - b
    k = 0
    while np.abs(rho_old) > tolerance and k < maxiter:

        #TODO : I added this test to avoid division by zero
        # if (s == 0):
          # s=1.0

        # print 'Marta score function = ', scorefunction(u / s, kind, clipping)
        #
        # # TODO : remove this
        # print 'Marta mean = ', np.mean(scorefunction(u / s, kind, clipping) * u / s)
        # if np.mean(scorefunction(u / s, kind, clipping) * u / s) == 0:
        #     print 'MARTA MEAN = 0 !'

        delta = rho_old / \
            np.mean(scorefunction(u / s, kind, clipping) * u / s) / s
        isqu = 1
        ok = 0
        while isqu < 30 and ok != 1:
            rho_new = np.mean(rhofunction(u / (s + delta), kind, clipping)) - b
            if np.abs(rho_new) < np.abs(rho_old):
                s = s + delta
                ok = 1
            else:
                delta /= 2
                isqu += 1
            if isqu == 30:
                # we tell it to stop, but we keep the iter for info
                maxiter = k
            rho_old = rho_new
            k += 1
    return np.abs(s)


# -------------------------------------------------------------------
# Looking for initial solutions
# -------------------------------------------------------------------
def getinitialsolution(y, a):
    import numpy as np
    import toolboxinverse_latest as inv
    import sys

    # line added to keep a constant initialx for testing purpose. remove this later
    #return np.array([-0.56076046, -2.96528342]).reshape(-1,1)

    # TODO : remove this line "return np.random.rand(a.shape[1])", only for testing purpose
    #return np.random.rand(a.shape[1])
    #return np.random.rand(a.shape[1])

    m = a.shape[0]  # getting dimensions
    n = a.shape[1]  # getting dimensions
    k = 0  # counting iterations
    while k < 100:
        perm = np.random.permutation(m)
        subsample = perm[0:n]  # random subsample
        ysubsample = y[subsample]  # random measurements
        asubsample = a[subsample, :]  # getting the rows
        r = np.linalg.matrix_rank(asubsample)

        # we assume that in these cases asubsample is well condtitioned
        if r == n:
            # use it to generate a solution
            initialx = inv.leastsquares(ysubsample, asubsample)
            return initialx
        else:
            k += 1
            if k == 100:
                # die gracefully
                sys.exit('I could not find initial solutions!')


# -------------------------------------------------------------------
# tau - scale ** 2
# -------------------------------------------------------------------
def tauscale(u, lossfunction, clipping, b, tolerance=1e-5):
    import numpy as np

    #TODO : uncomment this line and remove m=u.shape[0]
    # m, n = u.shape

    m = u.shape[0]

    mscale = mscaleestimator(u, tolerance, b, clipping[0], lossfunction)  # M scale

    # if mscale is zero, tauscale is zero as well
    if (mscale == 0):
      tscale = 0
    else:
        # (tau scale) ** 2
        tscale = mscale ** 2 * (1 / m) * np.sum(rhofunction(u / mscale, lossfunction, clipping[1]))
    return tscale

