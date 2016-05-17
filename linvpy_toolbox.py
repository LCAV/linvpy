from __future__ import division
import numpy as np
import linvpy as lp
import sys


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
    if lossfunction == 'optimal':  # weights for the rho tau.
        w = np.sum(2. * rhooptimal(u, clipping[1]) - scoreoptimal(u, clipping[1]) * u) \
            / np.sum(scoreoptimal(u, clipping[0]) * u)
    else:
        sys.exit('unknown type of loss function %s' %
                 lossfunction)  # die gracefully
    return w


# -------------------------------------------------------------------
# Optimal score function
# -------------------------------------------------------------------
def scoreoptimal(u, clipping):
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


#==== USED FOR REAL IN LINVPY !


# -------------------------------------------------------------------
# Looking for initial solutions
# -------------------------------------------------------------------
def getinitialsolution(y, a):
    import toolboxinverse as inv
    import sys
    # line added to keep a constant initialx for testing purpose. remove this later
    # return np.array([-0.56076046, -2.96528342]).reshape(-1,1)

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
            initialx = lp.least_squares(asubsample, ysubsample)
            return initialx
        else:
            k += 1
            if k == 100:
                # die gracefully
                sys.exit('I could not find initial solutions!')


def tauscale(u, lossfunction, clipping, b, tolerance=1e-5):
    m, n = u.shape
    mscale = mscaleestimator(
        u, tolerance, b, clipping, lossfunction)  # M scale
    tscale = mscale ** 2 * \
        (1 / m) * np.sum(rhofunction(u / mscale, lossfunction, clipping)
                         )  # (tau scale) ** 2
    return tscale


# -------------------------------------------------------------------
# M - scale estimator function
# -------------------------------------------------------------------
def mscaleestimator(u, tolerance, b, clipping, kind):
    maxiter = 100
    s = np.median(np.abs(u)) / .6745  # initial MAD estimation of the scale
    rho_old = np.mean(rhofunction(u / s, kind, clipping)) - b
    k = 0
    while np.abs(rho_old) > tolerance and k < maxiter:
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
# Rho functions
# -------------------------------------------------------------------
def rhofunction(u, kind, clipping):
    if kind == 'optimal':  # least squares
        r = rhooptimal(u, clipping)  # get the estimate
    else:  # unknown method
        sys.exit('unknown rho function %s' % kind)  # die gracefully
    return r  # return the score function that we need

# -------------------------------------------------------------------
# Score functions for the robust regressors
# -------------------------------------------------------------------


def scorefunction(u, kind, clipping):
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
        score = weighttau * \
            scoreoptimal(u, clipping[0]) + scoreoptimal(u, clipping[1])
    else:  # unknown method
        sys.exit('unknown method %s' % kind)  # die gracefully
    return score  # return the score function that we need


# -----------------------------------------------
# Optimal loss function (rho)
# -------------------------------------------------------------------
def rhooptimal(u, clipping):
    """
    The Fast-Tau Estimator for Regression, Matias SALIBIAN-BARRERA, Gert WILLEMS, and Ruben ZAMAR
    www.tandfonline.com/doi/pdf/10.1198/106186008X343785

    The equation is found p. 611. To get the exact formula, it is necessary to use 3*c instead of c.
    """
    y = np.abs(u / clipping)
    r = np.ones(u.shape)
    i = y <= 2.  # middle part of the curve
    r[i] = y[i] ** 2 / 2. / 3.25
    i = np.logical_and(y > 2, y <= 3)  # intermediate part of the curve
    f = lambda z: (1.792 - 0.972 * z ** 2 + 0.432 * z **
                   4 - 0.052 * z ** 6 + 0.002 * z ** 8) / 3.25
    r[i] = f(y[i])
    return r
