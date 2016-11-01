__author__ = 'GuillaumeBeaud'

import linvpy2 as lp
import numpy as np
from tests import generate_random as gen
from random import randint

random = gen.generate_random(5,6)




A = np.matrix([[1, 2], [3, 4], [3, 4]])
x = np.matrix([[7, 5, 4]])

my_estimator = lp.MEstimator(lp.Huber)
print 'first ', my_estimator.irls(random[0], random[1])

tik = lp.Tikhonov()
print tik.regularize(A, x, 0.2)

my_second_estimator = lp.MEstimator(loss_function=lp.Cauchy, clipping=0.1)
print 'second ', my_second_estimator.estimate(A,x)


print '=========================================== LIMIT ====================================='
# ============================================== ABOVE IS OK =====================================


loss_functions = [lp.Huber, lp.Bisquare, lp.Cauchy, lp.Optimal]

TESTING_ITERATIONS = 10


def test_MEstimator_ill_conditioned():
    for loss in loss_functions :
        print 'loss = ', loss
        m_estimator = lp.MEstimator(loss_function=loss, lamb=2.1) # creates an m-estimator with each of the loss functions
        for i in range(2, TESTING_ITERATIONS):
            # random (A,y) ill conditioned tuple with i rows
            print m_estimator.estimate(
                gen.generate_random_ill_conditioned(i)[0],
                gen.generate_random_ill_conditioned(i)[1].reshape(-1)
            )

test_MEstimator_ill_conditioned()

def test_tikhonov():
    tiko = lp.Tikhonov()
    for i in range(2, TESTING_ITERATIONS):
        # random (A,y) ill conditioned tuple with i rows
        print tiko.regularize(
            gen.generate_random_ill_conditioned(i)[0],
            gen.generate_random_ill_conditioned(i)[1].reshape(-1),
            lamb=randint(0,20)
        )

test_tikhonov()