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
			A,y = generate_random.generate_random(i,2)
			
			# clones the matrix A and vector y not to work on the same in memory
			y_gui = copy.deepcopy(y)
			a_gui = copy.deepcopy(A)

			y_gui2 = copy.deepcopy(y)
			a_gui2 = copy.deepcopy(A)

			y_marta = copy.deepcopy(y.reshape(-1,1))
			a_marta = copy.deepcopy(A)

			y_marta2 = copy.deepcopy(y.reshape(-1,1))
			a_marta2 = copy.deepcopy(A)

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

			xfinal_marta2, tscalefinal_marta2 = inv.fasttau(
			  y_marta2,
			  a_marta2,
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

			xfinal_lp2, tscalefinal_lp2 = lp.fasttau(
			  y_gui2,
			  a_gui2,
			  lossfunction,
			  clipping,
			  n_initial_solutions
			)

			# this test calls twice the same function with the same values to
			# check that two executions give exactly the same result.
			print "MARTA's fastau1 : ", xfinal_marta.reshape(-1), tscalefinal_marta
			print "MARTA's fastau2 : ", xfinal_marta2.reshape(-1), tscalefinal_marta2
			print "LinvPy's fastau1 : ", xfinal_lp.reshape(-1), tscalefinal_lp
			print "LinvPy's fastau1 : ", xfinal_lp2.reshape(-1), tscalefinal_lp2

			print "==================="

			# asserts that xhat's are equals and shat's are equals
			self.assertEquals(xfinal_lp.all(), xfinal_marta.all())
			self.assertEquals(tscalefinal_lp.all(), tscalefinal_marta.all())



if __name__ == '__main__':
	unittest.main()
