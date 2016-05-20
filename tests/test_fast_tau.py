from __future__ import division
import unittest
import numpy as np
import linvpy as lp
import generate_random
import optimal as opt
import random
import copy
import toolboxinverse as inv

TESTING_ITERATIONS = 5

class TestUM(unittest.TestCase):

	# Tests LinvPy's basictau against Marta's version
	def test_fast_tau(self):
		for i in range(4, TESTING_ITERATIONS):
			A,y = generate_random.generate_random(i,i)
			
			# clones the matrix A and vector y not to work on the same in memory
			x_gui = copy.deepcopy(y)
			a_gui = copy.deepcopy(A)
			y_gui = np.dot(a_gui, x_gui)

			x_marta = copy.deepcopy(y.reshape(-1,1))
			a_marta = copy.deepcopy(A)
			y_marta = np.dot(a_marta, x_marta)

			noise_vector = 0.5 * np.random.rand(y.size, 1)

			print "noise vector = ", noise_vector
			print "y marta = ", y_marta
			print "y guillaume = ", y_gui

			y_marta = y_marta + noise_vector
			y_gui = y_gui + noise_vector.reshape(-1)

			print "y marta2 = ", y_marta
			print "y guillaume2 = ", y_gui

			# random float clipping tuple between 0 and 10
			clipping = (random.uniform(0.1, 10.0), random.uniform(0.1, 10.0))
			
			# define parameters necessary for basic tau...
			lossfunction = 'optimal'

			# how many initial solutions do we want to try
			n_initial_solutions = random.randint(1,20)
			n_initial_solutions = 15

			# max number of iterations for irls
			max_iter = random.randint(1,100)

			# how many solutions do we keep
			n_best = random.randint(1,20)


			xfinal_marta, tscalefinal_marta = inv.fasttau(
			  y_marta,
			  a_marta,
			  lossfunction,
			  clipping,
			  n_initial_solutions
			)

			xfinal_lp, tscalefinal_lp = lp.fasttau(
			  y_gui,
			  a_gui,
			  lossfunction,
			  clipping,
			  n_initial_solutions
			)

			# this test calls twice the same function with the same values to
			# check that two executions give exactly the same result.
			print "MARTA's fastau : ", xfinal_marta.reshape(-1), tscalefinal_marta
			
			print "LinvPy's fastau : ", xfinal_lp.reshape(-1), tscalefinal_lp

			print "==================="

			# asserts that xhat's are equals and shat's are equals
			self.assertEquals(xfinal_lp.all(), xfinal_marta.all())
			self.assertEquals(tscalefinal_lp.all(), tscalefinal_marta.all())



if __name__ == '__main__':
	unittest.main()
