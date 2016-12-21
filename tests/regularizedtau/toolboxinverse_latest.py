"""
===================================================================
INVERSE PROBLEM
This toolbox contains all the functions related to solving the
inverse problem
Marta Martinez-Camara, EPFL
===================================================================
"""
from __future__ import division  # take the division operator from future versions


# -------------------------------------------------------------------
# Least squares with l2 regularization
# -------------------------------------------------------------------
def ridge(y, a, lmbd):
    import numpy as np
    m, n = a.shape
    xhat = np.linalg.solve(np.dot(a.T, a) + lmbd * np.eye(n), np.dot(a.T, y))  # estimate
    return xhat


# -------------------------------------------------------------------
# Least squares
# -------------------------------------------------------------------
def leastsquares(y, a):
    import numpy as np
    xhat = np.linalg.lstsq(a, y)[0]  # plain vanilla least squares
    return xhat


# -------------------------------------------------------------------
# Least squares with l1 regularization
# -------------------------------------------------------------------
# def lasso(y, a, lmbd, initialx, tolerance=1e-10, maxiter=100):  # we implement lasso with weights in the x
#   import numpy as np
#   y = np.array(y)  # cast to numpy array
#   x = np.array(initialx)  # initialize solution
#   xdis = 2 * tolerance  # minimum distance between two consecutive solutions
#   k = 0  # iterations counter initialization
#   steps = np.zeros((maxiter, 1))
#   while xdis > tolerance and k < maxiter:  # while we do not reach any stop condition
#     wlasso = 1 / x  # weights for lasso
#     if isinstance(wlasso, list):  # if the source is multidimensional
#       w = np.diag(wlasso)  # diagonal matrix
#     else:  # otherwise
#       w = wlasso
#     newx = np.linalg.solve(np.dot(a.T, a) + lmbd * np.dot(w.T, w), np.dot(a.T, y))  # new estimate
#     xdis = np.linalg.norm(x - newx)  # normalized distance
#     steps[k] = xdis  # step in this iteration. To check convergence
#     x = newx  # update x
#     k += 1  # update iter
#
#   return x, k, steps

# -------------------------------------------------------------------
# Least squares with l1 regularization using cvxpy
# -------------------------------------------------------------------
def lasso(y, a, lmbd):
    import cvxpy as cvx
    import numpy as np
    gamma = cvx.Parameter(sign="positive")
    measurementsize, sourcesize = a.shape
    # Construct the problem.
    xopt = cvx.Variable(sourcesize)
    error = cvx.sum_squares(a * xopt - y)
    obj = cvx.Minimize(error + gamma * cvx.norm(xopt, 1))
    prob = cvx.Problem(obj)
    gamma.value = lmbd
    try:
        prob.solve()  # solve the problem
        if prob.value == float('inf'):  # if infeasible
            xhat = np.zeros((sourcesize, 1))
            # xhat = np.nan
        else:
            xhat = np.array(xopt.value)
    except cvx.SolverError:  # if no solution found, we catch the error to keep going
        print 'Ooops! no solution found'
        xhat = np.zeros((sourcesize, 1))  # this is the boundary solution.

    return xhat


# -------------------------------------------------------------------
# Iterative Re-weighted Least Squares
# -------------------------------------------------------------------
def irls(y, a, kind, lossfunction, regularization, lmbd, initialx, initialscale, clipping, maxiter=101, tolerance=1e-5,
         b=0.5):
    """
    Iterative Re-weighted Least Squares algorithm

    Input arguments:
    y: measurements
    a: model matrix
    kind: type of method (M)
    lossfunction: type of loss function that we want (squared, huber)
    regularization: type of regularization. Options: none, l2
    initialx: initial solution
    initialscale: initial scale. For the M estimator the scale is fixed, so in this case this is the preliminary scale
    clipping: clipping parameter for the loss function
    lmbd: regularization parameter
    tolerance: if two consecutive iterations give two solutions closer than tolerance, the algorithm stops
    maxiter: maximum number of iterations. If the algorithm reaches this, it stops
    """
    import numpy as np
    import toolboxutilities_latest as util
    import sys
    # ------- Initialization -------------------------------
    m, n = a.shape  # number of measurements and number of unknowns
    y = np.array(y)  # cast to numpy array
    res = y - np.dot(a, initialx)  # initial residuals
    x = initialx  # initialize solution
    scale = initialscale  # for the moment, preliminary scale
    xdis = 2 * tolerance  # minimum distance between two consecutive solutions
    k = 0  # iterations counter initialization
    w = 0  # initialization
    steps = np.zeros((maxiter, 1))
    # ------- Iterating -----------------------------------------
    while xdis > tolerance and k < maxiter:  # while we do not reach any stop condition
        if kind == 'tau':
            scale *= np.sqrt(
                np.mean(util.rhofunction(res / scale, lossfunction, clipping[0])) / b)  # approximate M-scale
            #  update with respect to new residuals
        rhat = res / scale  # normalized residual
        w = util.weights(rhat, kind, lossfunction, clipping, m)  # getting the weights we need
        sqw = np.sqrt(w)  # to convert it to a matrix multiplication
        aw = a * sqw
        yw = y * sqw  # now these are the LS arguments

        # print 'marta aw, yw = ', aw, yw

        if regularization == 'none':
            newx = leastsquares(yw, aw)  # solving the weighted LS system
        elif regularization == 'l2':
            newx = ridge(yw, aw, lmbd)
        elif regularization == 'l1':
            newx = lasso(yw, aw, lmbd)
        else:
            sys.exit('unknown type of regularization' % regularization)  # die gracefully
        # if np.any(np.isnan(newx)):  # this is a catch because sometimes the algorithm diverges
        #   newx = x  # reset to the result of the last iteration

        print 'Marta x, newx = ', x, newx

        #TODO : xdis changed by Guillaume
        # Distance between previous and current iteration
        xdis = np.linalg.norm(np.subtract(x, newx))

        #xdis = np.linalg.norm(x - newx)  # normalized distance

        print 'Marta xdis = ', xdis

        steps[k] = xdis  # step in this iteration. To check convergence
        res = y - np.dot(a, newx)  # new residual
        x = newx  # update x
        k += 1  # update iter

        # TODO : added by Guillaume
        if (xdis < tolerance):
            return x, scale, k, w, steps

        # ---------- Wrapping up ----------------------------------------
    return x, scale, k, w, steps


# -------------------------------------------------------------------
# M - estimator
# -------------------------------------------------------------------
def mestimator(y, a, lossfunction, clipping, preliminaryscale):
    import numpy as np
    m, n = a.shape
    initialx = np.ones((n, 1))  # initial solution
    xhat, s, k, w, steps = \
        irls(y, a, 'M', lossfunction, 'none', 0, initialx, preliminaryscale, clipping)
    return xhat


# -------------------------------------------------------------------
# M - estimator with l2 regularization
# -------------------------------------------------------------------
def mridge(y, a, lossfunction, clipping, preliminaryscale, lmbd):
    import numpy as np
    m, n = a.shape
    initialx = np.ones((n, 1))  # initial solution
    xhat, s, k, w, steps = \
        irls(y, a, 'M', lossfunction, 'l2', lmbd, initialx, preliminaryscale, clipping)
    return xhat


# -------------------------------------------------------------------
# M - estimator with l1 regularization
# -------------------------------------------------------------------
def mlasso(y, a, lossfunction, clipping, preliminaryscale, lmbd):
    import numpy as np
    m, n = a.shape
    initialx = np.random.rand(n, 1)  # initial solution
    xhat, s, k, w, steps = \
        irls(y, a, 'M', lossfunction, 'l1', lmbd, initialx, preliminaryscale, clipping)
    return xhat


# -------------------------------------------------------------------
# basic tau - estimator
# -------------------------------------------------------------------
def basictau(y, a, lossfunction, clipping, ninitialx, maxiter=100, nmin=1, initialx=0, b=0.5):
    import numpy as np
    import toolboxutilities_latest as util
    mintauscale = np.empty((nmin, 1))
    mintauscale[:] = float("inf")  # initializing objective function with infinite
    k = 0  # iteration counter
    xhat = np.zeros((a.shape[1], nmin))  # to store the best nmin minima
    givenx = 0  # to know if we have predefined initial solutions or not
    if ninitialx == 0:  # if we introduce the initialx predefined
        ninitialx = initialx.shape[1]
        givenx = 1  # variable to know if there are given initial solutions
    while k < ninitialx:

        #TODO : for testing purpose we make initx deterministic to avoid random values
        initx = np.ones(a.shape[1])
        #initx = util.getinitialsolution(y, a)  # getting a new initial solution

        if givenx == 1:
            initx = np.expand_dims(initialx[:, k], axis=1)  # take the given initial solution instead

        print 'initx, A = ', initx, a
        print 'a dot initx = ', np.dot(a, initx)
        initialres = y - np.dot(a, initx)
        # TODO : initialres = np.array(initialres) added to make basictau work
        initialres = np.array(initialres)
        print 'initialres = ', initialres
        initials = np.median(np.abs(initialres)) / 0.6745  # initial MAD estimation of the scale

        print 'YAY'
        xhattmp, scaletmp, ni, w, steps = irls(y, a, 'tau', 'optimal', 'none', 0, initx, initials, clipping, maxiter)
        # getting the new local minimum
        res = y - np.dot(a, xhattmp)  # new residuals for the local minimum
        tscalesquare = util.tauscale(res, lossfunction, clipping[0], b)  # objective function in the local minimum
        k += 1  # update number of iterations
        if tscalesquare < np.amax(mintauscale):  # is it among the best nmin minima?
            mintauscale[np.argmax(mintauscale)] = tscalesquare  # keep it!
            xhat[:, np.argmax(mintauscale)] = np.squeeze(xhattmp)  # new global minimum

    return xhat, mintauscale


# -------------------------------------------------------------------
# basic tau - estimator with l2 regularization
# -------------------------------------------------------------------
def basictauridge(y, a, lossfunction, clipping, ninitialx, lmbd, maxiter=100, nmin=1, initialx=0, b=0.5):
    import numpy as np
    import toolboxutilities as util
    minobj = np.empty((nmin, 1))
    minobj[:] = float("inf")  # initializing objective function with infinite
    k = 0  # iteration counter
    xhat = np.zeros((a.shape[1], nmin))  # to store the best nmin minima
    givenx = 0  # to know if we have predefined initial solutions or not
    if ninitialx == 0:  # if we introduce the initialx predefined
        ninitialx = initialx.shape[1]
        givenx = 1  # variable to know if there are given initial solutions
    while k < ninitialx:
        initx = util.getinitialsolution(y, a)  # getting a new initial solution
        if givenx == 1:
            initx = np.expand_dims(initialx[:, k], axis=1)  # take the given initial solution instead
        initialres = y - np.dot(a, initx)
        initials = np.median(np.abs(initialres)) / .6745  # initial MAD estimation of the scale
        xhattmp, scaletmp, ni, w, steps = irls(y, a, 'tau', 'optimal', 'l2', lmbd, initx, initials, clipping, maxiter)
        # getting the new local minimum
        res = y - np.dot(a, xhattmp)  # new residuals for the local minimum
        tscalesquare = util.tauscale(res, lossfunction, clipping[0], b)  # tau scale squared in the local minimum
        k += 1  # update number of iterations
        obj = tscalesquare + lmbd * np.dot(xhattmp.T, xhattmp)  # objective function
        if obj < np.amax(minobj):  # is it the global minimum?
            minobj[np.argmax(minobj)] = obj  # new min objective function
            xhat[:, np.argmax(minobj)] = np.squeeze(xhattmp)  # new global minimum

    return xhat, minobj


# -------------------------------------------------------------------
# basic tau - estimator with l1 regularization
# -------------------------------------------------------------------
def basictaulasso(y, a, lossfunction, clipping, ninitialx, lmbd, maxiter=100, nmin=1, initialx=0, b=0.5):
    import numpy as np
    import toolboxutilities_latest as util
    minobj = np.empty((nmin, 1))
    minobj[:] = float("inf")  # initializing objective function with infinite
    k = 0  # iteration counter
    xhat = np.zeros((a.shape[1], nmin))  # to store the best nmin minima
    givenx = 0  # to know if we hif prob.value == float('inf'):  # if infeasible
    #     xhat = np.zeros((sourcesize, 1))
    #     #xhat = np.nan
    #   else:
    #     xhat = np.array(xopt.value)ave predefined initial solutions or not
    if ninitialx == 0:  # if we introduce the initialx predefined
        ninitialx = initialx.shape[1]
        givenx = 1  # variable to know if there are given initial solutions
    while k < ninitialx:
        initx = util.getinitialsolution(y, a)  # getting a new initial solution
        if givenx == 1:
            initx = np.expand_dims(initialx[:, k], axis=1)  # take the given initial solution instead
        initialres = y - np.dot(a, initx)
        initials = np.median(np.abs(initialres)) / .6745  # initial MAD estimation of the scale
        xhattmp, scaletmp, ni, w, steps = irls(y, a, 'tau', 'optimal', 'l1', lmbd, initx, initials, clipping,
                                               maxiter, 1e-10)  # getting
        # the new local minimum
        res = y - np.dot(a, xhattmp)  # new residuals for the local minimum
        tscalesquare = util.tauscale(res, lossfunction, clipping[0], b)  # tau scale squared in the local minimum
        k += 1  # update number of iterations
        obj = tscalesquare + lmbd * np.linalg.norm([xhattmp], 1)  # brackets for 1d function
        # obj = tscalesquare + lmbd * np.linalg.norm(xhattmp, 1)
        if obj < np.amax(minobj):  # is it the global minimum?
            minobj[np.argmax(minobj)] = obj  # new min objective function
            xhat[:, np.argmax(minobj)] = np.squeeze(xhattmp)  # new global minimum

    return xhat, minobj


# -------------------------------------------------------------------
# fast tau - estimator
# -------------------------------------------------------------------
def fasttau(y, a, lossfunction, clipping, ninitialx, nmin=5, initialiter=5):
    xhat, mintauscale = basictau(y, a, lossfunction, clipping, ninitialx, initialiter, nmin)  # first round: only
    # initialiter iterations. We keep the nmin best solutions
    xfinal, tauscalefinal = basictau(y, a, lossfunction, clipping, 0, 100, 1, xhat)  # iterate the best solutions
    # until convergence

    return xfinal, tauscalefinal


# -------------------------------------------------------------------
# fast tau - estimator with l2 regularization
# -------------------------------------------------------------------
def fasttauridge(y, a, lossfunction, clipping, ninitialx, lmbd, nmin=5, initialiter=5):
    xhat, minobj = basictauridge(y, a, lossfunction, clipping, ninitialx, lmbd, initialiter, nmin)  # first round: only
    # initialiter iterations. We keep the nmin best solutions
    xfinal, objfinal = basictauridge(y, a, lossfunction, clipping, 0, lmbd, 100, 1, xhat)  # iterate the best solutions
    # until convergence

    return xfinal, objfinal


# -------------------------------------------------------------------
# fast tau - estimator with l1 regularization
# -------------------------------------------------------------------
def fasttaulasso(y, a, lossfunction, clipping, ninitialx, lmbd, nmin=5, initialiter=5):
    xhat, minobj = basictaulasso(y, a, lossfunction, clipping, ninitialx, lmbd, initialiter, nmin)  # first round: only
    # initialiter iterations. We keep the nmin best solutions
    xfinal, objfinal = basictaulasso(y, a, lossfunction, clipping, 0, lmbd, 100, 1, xhat)  # iterate the best solutions
    # until convergence

    return xfinal, objfinal
