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
def lasso(a, y, lmbd):
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
      #xhat = np.nan
    else:
      xhat = np.array(xopt.value)
  except cvx.SolverError:  # if no solution found, we catch the error to keep going
    print 'Ooops! no solution found'
    xhat = np.zeros((sourcesize, 1))  # this is the boundary solution.

  return xhat


# -------------------------------------------------------------------
# Iterative Re-weighted Least Squares
# -------------------------------------------------------------------
def irls(y, a, kind, lossfunction, regularization, lmbd, initialx, initialscale, clipping, maxiter=100, tolerance=1e-5,
         b=0.5):
  """
  Iterative Re-weighted Least Squares algorithm

  Input arguments:
  y: vector y in y - Ax
  a: matrix a in y - Ax
  kind: type of method (M or tau)
  lossfunction: type of rho function that we want (squared, huber, optimal)
  regularization: type of regularization. Options: none, l2
  initialx: initial solution
  initialscale: initial scale.
  clipping: clipping parameter for the rho function. In the case of tau estimator, we need two values! (we have two rhos
            functions)
  lmbd: regularization parameter
  tolerance: if two consecutive iterations give two solutions closer than tolerance, the algorithm stops
  maxiter: maximum number of iterations. If the algorithm reaches this, it stops
  """
  import numpy as np
  import toolboxutilities as util
  import sys
  # ------- Initialization -------------------------------

  # number of measurements and number of unknowns
  m, n = a.shape

  # cast to numpy array
  y = np.array(y)

  # initial residuals
  res = y - np.dot(a, initialx)

  # initialize solution
  x = initialx

  # for the moment, preliminary scale
  scale = initialscale

  # minimum distance between two consecutive solutions
  xdis = 2 * tolerance

  # iterations counter initialization
  k = 0

  # initialization weights
  w = 0
  steps = np.zeros((maxiter, 1))

  # ------- Iterating -----------------------------------------
  while xdis > tolerance and k < maxiter:
    # while we do not reach any stop condition
    if kind == 'tau':
      # if we are computing the tau estimator, we need to upgrade the estimation of the scale in each iteration
      # approximation of the  M-scale

      scale *= np.sqrt(np.mean(util.rhofunction(res / scale, lossfunction, clipping[0])) / b)

    #  normalize residuals ((y - Ax)/ scale)
    rhat = res / scale

    #print "Marta's rhat = ", rhat

    # getting the weights we need (different if we have an M or a tau estimator)
    w = util.weights(rhat, kind, lossfunction, clipping, m)

    #print "Marta's weights matrix = ", w

    # once we have the weights, we solved the least squares problem
    sqw = np.sqrt(w)  # to convert it to a matrix multiplication
    #print "Marta's A matrix = ", a
    #print "Marta's square weight matrix = ", sqw
    aw = a * sqw
    yw = y * sqw  # now these are the LS arguments

    #print "Marta's a_weighted = ", aw

    # use the corresponding function, depending on the regularization
    if regularization == 'none':
      newx = leastsquares(yw, aw)  # solving the weighted LS system
    elif regularization == 'l2':
      newx = ridge(yw, aw, lmbd)
    elif regularization == 'l1':
      newx = lasso(aw, yw, lmbd)
    else:
       # die gracefully
       sys.exit('unknown type of regularization' % regularization)

    # if np.any(np.isnan(newx)):  # this is a catch because sometimes the algorithm diverges
    #   newx = x  # reset to the result of the last iteration

    # distance between the last x that we found and the new one. If they are too similar, we assume convergence and
    # we stop the algorithm (condition in while)
    xdis = np.linalg.norm(x - newx)

    # store the distance for further analysis and debugging
    steps[k] = xdis

    # new residual with the new x
    res = y - np.dot(a, newx)

    # update x
    x = newx

    # update counter
    k += 1
  # ---------- Wrapping up ----------------------------------------

  # return the estimate x, the scale, number of iterations k and distances between solutions (
  # to analyzse convergence later)
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
def basictau(y, a, lossfunction, clipping, ninitialx, maxiter=100, nbest=1, initialx=0, b=0.5):
    """
    This rutine minimazes the objective function associated with the tau-estimator.
    This function is hard to minimize because it is non-convex. This means that it has several local minima. Depending on
    the initial x that we use for our minimization, we will end up in a different local minimum (for the m-estimator is
    not like this; the function in that case is convex and we always arrive to it, independently of the initial solution)

    In this algorithm we take the 'brute force' approach: let's try many different initial solutions, and let's pick the
    minimum with smallest value. The output of basictau are the best nbest minima (we will need them later)

    :param y: vector y in y - Ax
    :param a: matrix A in y - Ax
    :param lossfunction: type of the rho function we are using
    :param clipping: clipping parameters. In this case we need two, because the rho function for the tau is composed two rho functions.
    :param ninitialx: how many different solutions do we want to use to find the global minimum (this function is not convex!)
                      if ninitialx=0, means the user introduced a predefined initial solution
    :param maxiter: maximum number of iteration for the irls algorithm
    :param nbest: we return the best nbest solutions. This will be necessary for the fast algorithm
    :param initialx: the user can define here the initial x he wants
    :param b: this is a parameter to estimate the scale

    :return xhat: contains the best nmin estimations of x
    :return mintauscale: value of the objective function when x = xhat
    """

    import numpy as np
    import toolboxutilities as util

    # to store the minimum values of the objective function (in this case is
    # the scale)
    mintauscale = np.empty((nbest, 1))

    # initializing objective function with infinite. When we have a x that gives a smaller value for the obj. function,
    # we store the value of the objective function here
    mintauscale[:] = float("inf")

    # count how many initial solutions are we trying
    k = 0

    # store here the best xhat (nbest of them)
    xhat = np.zeros((a.shape[1], nbest))  # to store the best nmin minima

    # auxiliary variable to check if the user introduced a predefined initial solution.
    # = 0 if we do not have initial x. =1 if we have a given initial x
    givenx = 0

    if (initialx == None) :
      initialx = np.ones(a.shape[1])

    if ninitialx == 0:
        # we have a predefined initial x
        ninitialx = initialx.shape[1]
        # set givenx to 1
        givenx = 1

    while k < ninitialx:
        # if still we did not reach the number of initial solutions that we want to try,
        # get a new initial solution initx (randomly)
        initx = util.get_initial_solution(y, a)

        if givenx == 1:
            # if we have a given initial solution initx, we take it
            initx = np.expand_dims(initialx[:, k], axis=1)

        # compute the residual y - Ainitx
        initialres = y - np.dot(a, initx)

        # estimate the scale using initialres
        initials = np.median(np.abs(initialres)) / .6745

        # solve irls using y, a, the tau weights, initx and initals. We get an
        # estimation of x, xhattmp
        xhattmp, scaletmp, ni, w, steps = irls(
            y, a, 'tau', 'optimal', 'none', 0, initx, initials, clipping, maxiter)

        # compute the value of the objective function using xhattmp
        # we compute the res first
        res = y - np.dot(a, xhattmp)

        # Value of the objective function using xhattmp
        tscalesquare = util.tauscale(res, lossfunction, clipping[0], b)

        # update counter
        k += 1

        # we checks if the objective function has a smaller value that then
        # ones we found before
        if tscalesquare < np.amax(mintauscale):
            # it is smaller, so we keep it!
            # store value for the objective function
            mintauscale[np.argmax(mintauscale)] = tscalesquare

            # store value of xhat
            xhat[:, np.argmax(mintauscale)] = np.squeeze(xhattmp)

    # we return the best solutions we found, with the value of the objective
    # function associated with the xhats
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
    initx = util.get_initial_solution(y, a)  # getting a new initial solution
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
  import toolboxutilities as util
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
    initx = util.get_initial_solution(y, a)  # getting a new initial solution
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
    #obj = tscalesquare + lmbd * np.linalg.norm(xhattmp, 1)
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
