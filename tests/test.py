__author__ = 'GuillaumeBeaud'

import linvpy as lp

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

print "psi =" , huber.psi(0.0), huber.psi(1.0), huber.psi(2.0), huber.psi(3.0)

y = np.array([1.0, 2.0, 3.0])

print huber.psi(y)

a = np.matrix([[1, 2], [3, 4], [5, 6]])

print huber.psi(a)


print "rho =" , huber.rho(1.0), huber.rho(2.0), huber.rho(3.0)

y = np.array([1, 2, 3])
print huber.rho(y)

a = np.matrix([[1, 2], [3, 4], [5, 6]])

print huber.rho(a)


a = np.matrix([[1, 2], [3, 4], [5, 6]])
y = np.array([1, 2, 3])

tau = lp.TauEstimator()

m = lp.MEstimator()

print tau.estimate(a,y)
print m.estimate(a,y)