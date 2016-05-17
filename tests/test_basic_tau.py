from __future__ import division
import unittest
import numpy as np
import linvpy as lp
import generate_random
import optimal as opt
import random
import copy
import toolboxinverse as inv

TESTING_ITERATIONS = 7

class TestUM(unittest.TestCase):

	# Tests LinvPy's basic tau against Marta's version
	def test_tau(self):
		for i in range(4, TESTING_ITERATIONS):
			A,y = generate_random.generate_random(i,2)
			
			# clones the matrix A and vector y not to work on the same in memory
			y_gui = copy.deepcopy(y)
			a_gui = copy.deepcopy(A)

			y_marta = copy.deepcopy(y.reshape(-1,1))
			a_marta = copy.deepcopy(A)

			# random float clipping tuple between 0 and 10
			clipping = (random.uniform(0.1, 10.0), random.uniform(0.1, 10.0))
			
			# define parameters necessary for basic tau...
			lossfunction = 'optimal'

			# how many initial solutions do we want to try
			n_initial_solutions = random.randint(1,20)
			n_initial_solutions = 15

			# max number of iterations for irls
			max_iter = random.randint(1,100)
			max_iter = 100

			# how many solutions do we keep
			n_best = random.randint(1,20)
			n_best = 3

			# calls Marta's basic tau estimator
			xhat_marta, shat_marta = inv.basictau(
			  y_marta,
			  a_marta,
			  lossfunction,
			  clipping,
			  n_initial_solutions,
			  max_iter,
			  n_best
			)

			# calls linvpy's basic tau estimator
			xhat_lp, shat_lp = lp.basictau(
			  y_gui,
			  a_gui,
			  lossfunction,
			  clipping,
			  n_initial_solutions,
			  max_iter,
			  n_best
			)

			print "Marta's output for tau-estimator = ", xhat_marta.reshape(-1), shat_marta.reshape(-1)
			print "LinvPy's output for tau-estimator = ", xhat_lp.reshape(-1), shat_lp.reshape(-1)

			print "==================="


			# asserts that xhat's are equals and shat's are equals
			self.assertEquals(xhat_lp.all(), xhat_marta.all())
			self.assertEquals(shat_lp.all(), shat_marta.all())



if __name__ == '__main__':
	unittest.main()
