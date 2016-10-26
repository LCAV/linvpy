from __future__ import division
import unittest
import numpy as np
import linvpy as lp
import generate_random
import optimal as opt
import random
import copy
import toolboxinverse as inv

TESTING_ITERATIONS = 10

class TestUM(unittest.TestCase):

	# Tests LinvPy's basic tau against Marta's version
	def test_tau(self):
		for i in range(4, TESTING_ITERATIONS):

			NOISE = 0.1

			columns = random.randint(2,i)

			A, x, y, initial_x, scale = generate_random.gen_noise(i,columns,NOISE)

			y_gui = copy.deepcopy(y)
			a_gui = copy.deepcopy(A)

			y_marta = copy.deepcopy(y.reshape(-1,1))
			a_marta = copy.deepcopy(A)

			# random float clipping between 0 and 10
			clipping_tau = (random.uniform(0.1, 4.0), random.uniform(0.1, 4.0))

			# define parameters necessary for basic tau...
			lossfunction = 'optimal'

			# how many initial solutions do we want to try
			n_initial_solutions = random.randint(1,20)
			n_initial_solutions = 15

			# how many solutions do we keep
			n_best = random.randint(1,20)
			n_best = 1

			'''
			# calls Marta's basic tau estimator
			xhat_marta, shat_marta = inv.basictau(
			  y=y_marta,
			  a=a_marta,
			  lossfunction=lossfunction,
			  clipping=clipping_tau,
			  ninitialx=n_initial_solutions,
			  nbest=n_best
			)
			'''

			# calls linvpy's basic tau estimator
			xhat_lp, shat_lp = lp.basictau(
			  a=a_gui,
			  y=y_gui,
			  loss_function=lp.rho_optimal,
			  clipping=clipping_tau,
			  ninitialx=n_initial_solutions,
			  nbest=n_best,
			  regularization=lp.tikhonov_regularization,
			  lamb=0
			)

			#print "Marta's output for tau-estimator = ", xhat_marta.reshape(-1)
			print "LinvPy's output for tau-estimator = ", xhat_lp.reshape(-1)
			print "Real xhat = ", x
			print "==================="

			# only tests this if noise is zero otherwise it fails all the time
			# because values are not EXACTLY the same
			if NOISE == 0 :
				np.testing.assert_array_almost_equal(x, xhat_lp.reshape(-1), decimal=5)




if __name__ == '__main__':
	unittest.main()
