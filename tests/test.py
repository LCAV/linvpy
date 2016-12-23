__author__ = 'GuillaumeBeaud'


# print '=========================================== LIMIT ====================================='


# ============================================== ABOVE IS OK =====================================
# ============================================== DEMO =====================================

import numpy as np
import linvpy as lp

a = np.matrix([[1, 2], [3, 4], [5, 6]])
y = np.array([1, 2, 3])

# Define your own loss function
class CustomLoss(lp.LossFunction):

    # Set your custom clipping
    def __init__(self, clipping=1.5):
        lp.LossFunction.__init__(self, clipping)
        if clipping is None:
            self.clipping = 0.7

    # Define your rho function : you can copy paste this and just change what's
    # inside the unit_rho
    def rho(self, array):
        # rho function of your loss function on ONE single element
        def unit_rho(element):
            # Simply return clipping * element for example
            return element + self.clipping
        # Vectorize the function
        vfunc = np.vectorize(unit_rho)
        return vfunc(array)

    # Define your psi function as the derivative of the rho function : you can
    # copy paste this and just change what's inside the unit_rho
    def psi(self, array):
        # rho function of your loss function on ONE single element
        def unit_psi(element):
            # Simply return the clipping for example
            return 1
        # Vectorize the function
        vfunc = np.vectorize(unit_psi)
        return vfunc(array)

custom_tau = lp.TauEstimator(loss_function=CustomLoss)
print custom_tau.estimate(a,y)


# Define your own regularization
class CustomRegularization(lp.Regularization):
    pass
    # Define your regularization function here
    def regularize(self, a, y, lamb=0):
        return np.ones(a.shape[1])

# Create your custom tau estimator with custom loss and regularization functions
# Pay attenation to pass the loss function as a REFERENCE (without the "()"
# after the name, and the regularization as an OBJECT, i.e. with the "()").
custom_tau = lp.TauEstimator(regularization=CustomRegularization())
print custom_tau.estimate(a,y)