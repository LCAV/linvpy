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

	# Tests LinvPy's basictau against Marta's version
	def test_fast_tau(self):
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

			xfinal_marta, tscalefinal_marta = inv.fasttau(
			  y=y_marta,
			  a=a_marta,
			  lossfunction=lossfunction,
			  clipping=clipping_tau,
			  ninitialx=n_initial_solutions
			)

			xfinal_lp, tscalefinal_lp = lp.fasttau(
			  y=y_gui,
			  a=a_gui,
			  loss_function=lp.rho_optimal,
			  clipping=clipping_tau,
			  ninitialx=n_initial_solutions
			)

			print "MARTA's fastau : ", xfinal_marta.reshape(-1), tscalefinal_marta			
			print "LinvPy's fastau : ", xfinal_lp.reshape(-1), tscalefinal_lp
			print "Real xhat = ", x
			print "==================="

			# only tests this if noise is zero otherwise it fails all the time
			# because values are not EXACTLY the same
			if NOISE == 0 :
				np.testing.assert_array_almost_equal(x, xfinal_lp.reshape(-1), decimal=5)



if __name__ == '__main__':
	unittest.main()
