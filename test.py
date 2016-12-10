__author__ = 'GuillaumeBeaud'

import linvpy as lp
import numpy as np
from tests import generate_random as gen
from random import randint
import random
import toolboxutilities as util
from tests import toolboxinverse as inv
import copy
from regularizedtau import toolboxutilities_latest as util_l
from regularizedtau import toolboxinverse_latest as inverse_l
from regularizedtau import linvpy_latest as lp_l

# random1 = gen.generate_random(5, 6)

TESTING_ITERATIONS = 30

LOSS_FUNCTIONS = [lp.Huber, lp.Bisquare, lp.Cauchy, lp.Optimal]  # reference to loss classes, not instances
#


# print '=========================================== LIMIT ====================================='


# ============================================== ABOVE IS OK =====================================
# ============================================== DEMO =====================================

import numpy as np
import linvpy as lp

A = np.matrix([[2, 2], [3, 4], [7, 6]])
y = np.array([1, 4, 3])

# create an instance of Tau, don't need to give any parameter
my_tau = lp.TauEstimator()


# or you can give one, two, three... or all parameters :
my_other_tau = lp.TauEstimator(
    loss_function=lp.Optimal,
    clipping_1=0.6,
    clipping_2=1.5,
    lamb=3,
    scale=1.5,
    b=0.7,
    tolerance=1e4, )

# creates an instance of M-estimator
my_m = lp.MEstimator()

# the estimate function returns the result of the corresponding estimator
print(my_tau.estimate(A, y))

print (my_other_tau.estimate(A, y))

print (my_m.estimate(A, y))

# fast tau
print (my_tau.fast_estimate(A, y))

# to change the clipping or any other parameter of the estimator :
my_tau.loss_function_1.clipping = 0.7
my_tau.tolerance = 1e3

# to create another Tau with another loss function :
my_tau_2 = lp.TauEstimator(loss_function=lp.Cauchy)

# change some parameter afterwards :
my_tau_2.lamb = 3
my_tau_2.b = 0.7
my_tau_2.tolerance = 1e4

# running with an initial solution :
x = np.array([5, 6])
print (my_tau_2.estimate(A, y, initial_x=x))


huber = lp.Huber()
print "rho =", huber.rho(15)

y = np.array([1, 2, 3])
print huber.psi(y)
a = np.matrix([[1, 2], [3, 4], [5, 6]])

print huber.rho(a)

