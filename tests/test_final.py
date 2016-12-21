__author__ = 'GuillaumeBeaud'

import linvpy as lp
import numpy as np
import random
from random import randint
from tests import generate_random as gen
from regularizedtau import toolboxutilities as util
from regularizedtau import toolboxutilities_latest as util_l
from regularizedtau import linvpy_latest as lp_l

# ===================================== DEFINITIONS ===================================

TESTING_ITERATIONS = 20

LOSS_FUNCTIONS = [lp.Huber, lp.Bisquare, lp.Cauchy, lp.Optimal]  # references to loss classes, not instances


# ===================================== TESTS ====================================

# sets the print precision to 20 decimals
np.set_printoptions(precision=10)

def plot_loss_functions(interval):
    for loss in LOSS_FUNCTIONS:
        loss = loss()  # instanciates the loss functions
        loss.plot(interval)


def test_MEstimator():
    for loss in LOSS_FUNCTIONS:
        m_estimator = lp.MEstimator(loss_function=loss)  # creates an m-estimator with each of the loss functions
        for i in range(2, TESTING_ITERATIONS):
            # random (A,y) tuple with i rows and A has a random number of columns between i and i+100
            m_estimator.estimate(
                np.random.rand(i, i + randint(0, 100)),
                np.random.rand(i).reshape(-1)
            )


def test_M_weights():
    toolbox_losses = ['huber', 'optimal']
    lp_losses = [lp.Huber, lp.Optimal]

    for i in range(0, 2):
        A = np.random.rand(randint(1, 10), randint(1, 10))
        clipping = np.random.uniform(0.1, 5)

        # creates an instance of the loss function with the current clipping
        my_loss = lp_losses[i](clipping=clipping)

        uw = util.weights(A, 'M', toolbox_losses[i], clipping, None)
        lw = my_loss.m_weights(A)

        np.testing.assert_allclose(uw, lw)


def test_MEstimator_ill_conditioned():
    for loss in LOSS_FUNCTIONS:
        m_estimator = lp.MEstimator(loss_function=loss)  # creates an m-estimator with each of the loss functions
        for i in range(2, TESTING_ITERATIONS):
            # random (A,y) ill conditioned tuple with i rows
            m_estimator.estimate(
                gen.generate_random_ill_conditioned(i)[0],
                gen.generate_random_ill_conditioned(i)[1].reshape(-1)
            )


# this is a crash test that checks that the function never crashes, not a value test
def test_tikhonov():
    tiko = lp.Tikhonov()
    for i in range(2, TESTING_ITERATIONS):
        # random (A,y) ill conditioned tuple with i rows
        tiko.regularize(
            gen.generate_random_ill_conditioned(i)[0],
            gen.generate_random_ill_conditioned(i)[1].reshape(-1),
            lamb=randint(0, 20)
        )


# tests the rho_optimal and psi_optimal of LinvPy VS rhooptimal and scoreoptimal of toolbox
def test_Optimal():
    for i in range(2, TESTING_ITERATIONS):
        # random clipping between 0.1 and 5
        CLIPPING = np.random.uniform(0.1, 5)

        # creates an instance of lp.Optimal
        opt = lp.Optimal(clipping=CLIPPING)

        # generates a random vector of size between 0 and 100
        y = np.random.rand(randint(1, 100))

        # optimal rho function of toolbox and optimal rho function of LinvPy
        rho_util = util.rhooptimal(np.asarray(y), CLIPPING)
        rho_lp = opt.rho(y)

        # optimal psi function of toolbox and optimal psi function of LinvPy
        psi_util = util.scoreoptimal(np.asarray(y), CLIPPING)
        psi_lp = opt.psi(y)

        # returns an error if the toolbox's rhooptimal and lp.Optimal.rho() are not equal
        np.testing.assert_allclose(rho_lp, rho_util)

        # returns an error if the toolbox's scoreoptimal and lp.Optimal.psi() are not equal
        np.testing.assert_allclose(psi_lp, psi_util)


# tests the scorefunction of LinvPy VS scorefunction of toolbox
def test_scorefunction():
    for i in range(2, TESTING_ITERATIONS):
        # CLIPPINGS = two random numbers between 0.1 and 5
        CLIPPINGS = (np.random.uniform(0.1, 5), np.random.uniform(0.1, 5))

        # creates an instance of tau estimator with the two random clippings
        tau = lp.TauEstimator(clipping_1=CLIPPINGS[0], clipping_2=CLIPPINGS[1], loss_function=lp.Optimal)

        # y = random vector of size between 0 and 100
        y = np.random.rand(randint(1, 100))

        # toolbox's scorefunction
        score_util = util.scorefunction(np.asarray(y), 'tau', CLIPPINGS)

        # linvpy's scorefunction
        score_lp = tau.score_function(y)

        # returns an error if the toolbox's scorefunction and lp's scorefunction are not equal
        np.testing.assert_allclose(score_lp, score_util)


# tests linvpy's mscale VS toolbox mscale
def test_mscale():
    for i in range(2, TESTING_ITERATIONS):
        # generates a random clipping between 0.1 and 5
        CLIPPING = np.random.uniform(0.1, 5)

        # creates an instance of TauEstimator
        tau = lp.TauEstimator(clipping_1=CLIPPING, clipping_2=CLIPPING, loss_function=lp.Optimal)

        # generates a random vector of size between 0 and 100
        y = np.random.rand(randint(1, 100))

        # computes the mscale for linvpy and toolbox
        linvpy_scale = tau.m_scale(y)
        toolbox_scale = util.mscaleestimator(u=y, tolerance=1e-5, b=0.5, clipping=CLIPPING, kind='optimal')

        # verifies that both results are the same
        assert toolbox_scale == linvpy_scale


def test_tau_scale():
    for i in range(2, TESTING_ITERATIONS):
        # generates random clipping between 0.1 and 5
        clipping_1 = np.random.uniform(0.1, 5)
        clipping_2 = np.random.uniform(0.1, 5)

        # generates a random vector of size between 0 and 100
        x = np.random.rand(randint(1, 100))

        my_tau = lp.TauEstimator(loss_function=lp.Optimal, clipping_1=clipping_1, clipping_2=clipping_2)

        linvpy_t = my_tau.tau_scale(x)
        util_t = util_l.tauscale(x, lossfunction='optimal', b=0.5, clipping=(clipping_1, clipping_2))

        np.testing.assert_allclose(linvpy_t, util_t)


def test_M_estimator_VS_Marta():
    for i in range(3, TESTING_ITERATIONS):
        NOISE = np.random.uniform(0, 1.0)
        # NOISE = 0
        # lamb = np.random.uniform(0,1.0)
        lamb = 0
        clipping = np.random.uniform(0.1, 5)

        A, x, y, initial_vector, initial_scale = gen.gen_noise(i, i, NOISE)

        xhat_marta = lp_l.irls(
            matrix_a=A,
            vector_y=y,
            loss_function='huber',
            kind='M',
            regularization=lp_l.tikhonov_regularization,
            lamb=lamb,
            initial_x=initial_vector.reshape(-1, 1),
            scale=initial_scale,
            clipping=clipping)

        my_m = lp.MEstimator(clipping=clipping,
                             loss_function=lp.Huber,
                             scale=initial_scale,
                             lamb=lamb)

        xhat_linvpy = my_m.estimate(A, y, initial_x=initial_vector)

        # print 'xhat marta = ', xhat_marta
        # print 'xhat linvpy = ', xhat_linvpy
        # print 'real x = ', x
        # very robust test; passes sometimes and sometimes not (a difference of 0.0000001 makes it fail)
        np.testing.assert_allclose(xhat_linvpy, xhat_marta)
        # print '=========================================='


# This test checks LinvPy2.0's tau estimator on all possible inputs and verify there's no error.
# NB: this DOES NOT test the mathematical correctness of outputs, it only tests that TauEstimator()
# can handle any types of inputs without crashing.
# For mathematical correctness, see test_TauEstimator_VS_Marta()
def test_TauEstimator_alone():
    for i in range(2, TESTING_ITERATIONS):

        # tests all regularizations
        for reg in (lp.Tikhonov(), lp.Lasso()):
        # tests all loss functions
            for loss in LOSS_FUNCTIONS:

                # intiates random inputs
                lamb = randint(0, 20)
                c1 = np.random.uniform(0.1, 5)
                c2 = np.random.uniform(0.1, 5)

                # clippings are randomly chosen between a random number or None with predominance for number
                clipping_1 = random.choice([c1, c1, c1, None])
                clipping_2 = random.choice([c2, c2, c2, None])

                # creates a tau instance
                tau_estimator = lp.TauEstimator(
                    loss_function=loss,
                    regularization=reg,
                    lamb=lamb,
                    clipping_1=clipping_1,
                    clipping_2=clipping_2)  # creates a tau-estimator with each of the loss functions

                # random (A,y) tuple with i rows and A has a random number of columns between i and i+100
                tau_estimator.estimate(
                    # A=np.random.rand(i, i + randint(0, 100)),
                    a=np.random.rand(i, i + randint(0, 100)),
                    y=np.random.rand(i).reshape(-1)
                )


def test_score_function_is_odd():
    for loss in LOSS_FUNCTIONS:

        my_tau = lp.TauEstimator(loss_function=loss)

        # print 'loss = ', loss

        for i in range(2, TESTING_ITERATIONS):

            # generates a random vector of size i with negative and positive values
            y = np.random.randn(100)

            score = my_tau.score_function(y)

            # print y, score

            for i in range(0, score.__len__()):
                assert np.sign(score[i]) == np.sign(y[i])

def test_TauEstimator_VS_Marta():
    for i in range(2, TESTING_ITERATIONS):
        # generates random clipping between 0.1 and 5
        clipping_1 = np.random.uniform(0.1, 5)
        clipping_2 = np.random.uniform(0.1, 5)

        # generates a random n_initial_x
        n_initial_x = 1

        # generates a random matrix of size i x i + random(0,100)
        A = np.random.rand(i, i + randint(0, 10))

        # generates a random vector of size i
        y = np.random.rand(i)

        my_tau = lp.TauEstimator(loss_function=lp.Optimal, clipping_1=clipping_1, clipping_2=clipping_2)

        linvpy_output = my_tau.estimate(a=A, y=y)

        marta_t = lp_l.basictau(
            a=A,
            y=np.matrix(y),
            loss_function='optimal',
            b=0.5,
            clipping=(clipping_1, clipping_2),
            ninitialx=n_initial_x
        )

        # print 'LinvPy Tau result = ', linvpy_output
        # print 'Marta Tau result = ', marta_t
        # print '========================'
        # print '========================'
        # print '========================'
        # print '========================'

        # asserts xhat are the same
        np.testing.assert_allclose(linvpy_output[0].reshape(-1, 1), marta_t[0])

# test that simply covers the fast_tau for code coverage purpose
def cover_fast_tau():
    my_tau = lp.TauEstimator()
    A = np.matrix([[2, 2], [3, 4], [7, 6]])
    y = np.array([1, 4, 3])
    my_tau.fast_estimate(A, y)

# ===================================== MAIN ==================================


# plot_loss_functions(15)

test_TauEstimator_alone()

test_TauEstimator_VS_Marta()

test_M_weights()

test_MEstimator()

test_MEstimator_ill_conditioned()

test_tikhonov()

test_Optimal()

test_scorefunction()

test_mscale()

test_tau_scale()

test_M_estimator_VS_Marta()

test_score_function_is_odd()

cover_fast_tau()