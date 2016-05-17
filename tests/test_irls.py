from __future__ import division
import unittest
import numpy as np
import linvpy as lp
import generate_random
import optimal as opt
import random
import copy
import toolboxinverse as inv

TESTING_ITERATIONS = 50
# For a matrix to be ill-conditioned, its condition number must be equal to or
# greather than ILL_CONDITION_CRITERIA
ILL_CONDITION_CRITERIA = 1000

PLOT_INTERVAL = 100

class TestUM(unittest.TestCase):

	# preparing to test
	def setUp(self):
		''' Setting up for the test '''
		#print 'FooTest:setUp_:end'
	 
	# ending the test
	def tearDown(self):
		'''Cleaning up after the test'''
		#print 'FooTest:tearDown_:begin'
		## do something...
		#print 'FooTest:tearDown_:end'
 

	# Tests LinvPy's m-estimator against Marta's version
	def test_mestimator(self):
		for i in range(2, TESTING_ITERATIONS):
			A,y = generate_random.generate_random(i,i)

			y_gui = copy.deepcopy(y)
			a_gui = copy.deepcopy(A)

			y_marta = copy.deepcopy(y.reshape(-1,1))
			a_marta = copy.deepcopy(A)

			# random float clipping between 0 and 10
			clipping_test = random.uniform(0.0, 10.0)

			xhat_linvpy = lp.irls(
				a_gui,
				y_gui,
				lp.psi_huber,
				clipping=clipping_test)

			xhat_marta = inv.irls(
				y=y_marta,
				a=a_marta,
				kind='tau',
				lossfunction='huber',
				regularization='none',
				lmbd=0,
				clipping=clipping_test)[0]

			#print "Marta's xhat for m-estimator = ", xhat_marta
			#print "LinvPy's xhat for m-estimator = ", xhat_linvpy

			self.assertEquals(xhat_linvpy.all(), xhat_marta.all())


if __name__ == '__main__':
	unittest.main()
