import numpy as np


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


# -------------------------------------------------------------------
# Optimal score function (psi)
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