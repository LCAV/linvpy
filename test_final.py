__author__ = 'GuillaumeBeaud'

import linvpy2 as lp
import numpy as np
from random import randint
from tests import generate_random as gen

# ===================================== DEFINITIONS =====================================

TESTING_ITERATIONS = 100

LOSS_FUNCTIONS = [lp.Huber, lp.Bisquare, lp.Cauchy, lp.Optimal] # reference to loss classes, not instances

# ===================================== TESTS =====================================


def plot_loss_functions(interval):
    for loss in LOSS_FUNCTIONS:
        loss = loss() # instanciates the loss functions
        print loss.rho(2)
        print loss.psi(3)
        loss.plot(interval)
        print loss.m_weights(1)


def test_MEstimator():
    for loss in LOSS_FUNCTIONS :
        print 'loss = ', loss
        m_estimator = lp.MEstimator(loss_function=loss) # creates an m-estimator with each of the loss functions
        for i in range(2, TESTING_ITERATIONS):
            # random (A,y) tuple with i rows and A has a random number of columns between i and i+100
            print m_estimator.estimate(
                np.random.rand(i, i+randint(0,100)),
                np.random.rand(i).reshape(-1)
            )

def test_MEstimator_ill_conditioned():
    for loss in LOSS_FUNCTIONS :
        print 'loss = ', loss
        m_estimator = lp.MEstimator(loss_function=loss) # creates an m-estimator with each of the loss functions
        for i in range(2, TESTING_ITERATIONS):
            # random (A,y) ill conditioned tuple with i rows
            print m_estimator.estimate(
                gen.generate_random_ill_conditioned(i)[0],
                gen.generate_random_ill_conditioned(i)[1].reshape(-1)
            )


def test_tikhonov():
    tiko = lp.Tikhonov()
    for i in range(2, TESTING_ITERATIONS):
        # random (A,y) ill conditioned tuple with i rows
        print tiko.regularize(
            gen.generate_random_ill_conditioned(i)[0],
            gen.generate_random_ill_conditioned(i)[1].reshape(-1),
            lamb=randint(0,20)
        )

# ===================================== MAIN =====================================


# plot_loss_functions(20)

test_MEstimator()

test_MEstimator_ill_conditioned()

test_tikhonov()