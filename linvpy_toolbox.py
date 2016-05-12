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
    score = weighttau * scoreoptimal(u, clipping[0]) + scoreoptimal(u, clipping[1])
  else:  # unknown method
    sys.exit('unknown method %s' % kind)  # die gracefully
  return score  # return the score function that we need



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
    if lossfunction == 'optimal':  # weights for the rho tau.
      w = np.sum(2. * rhooptimal(u, clipping[1]) - scoreoptimal(u, clipping[1]) * u) \
          / np.sum(scoreoptimal(u, clipping[0]) * u)
    else:
      sys.exit('unknown type of loss function %s' % lossfunction)  # die gracefully
    return w


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
  y = np.abs(u / clipping)
  r = np.ones(u.shape)
  i = y <= 2.  # middle part of the curve
  r[i] = y[i] ** 2 / 2. / 3.25
  i = np.logical_and(y > 2, y <= 3)  # intermediate part of the curve
  f = lambda z: (1.792 - 0.972 * z ** 2 + 0.432 * z ** 4 - 0.052 * z ** 6 + 0.002 * z ** 8) / 3.25
  r[i] = f(y[i])
  return r
