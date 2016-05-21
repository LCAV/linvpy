from __future__ import division
import unittest
import numpy as np
import linvpy as lp
import generate_random
import random
import copy
import toolboxinverse as inv

TESTING_ITERATIONS = 10
# For a matrix to be ill-conditioned, its condition number must be equal to or
# greather than ILL_CONDITION_CRITERIA

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
	def test_irls(self):
		for i in range(5, TESTING_ITERATIONS):

			NOISE = 0

			columns = random.randint(2,i)

			A,x,y,initial_vector,initial_scale = generate_random.gen_noise(i,columns, NOISE)

			y_gui = copy.deepcopy(y)
			a_gui = copy.deepcopy(A)

			y_marta = copy.deepcopy(y.reshape(-1,1))
			a_marta = copy.deepcopy(A)

			# random float clipping between 0 and 10
			clipping_single = random.uniform(0.1, 4.0)

			test_kind='M'

			lamb=0.1

			'''
			xhat_marta = inv.irls(
				y=y_marta,
				a=a_marta,
				kind=test_kind,
				lossfunction='huber',
				regularization='l1',
				lmbd=lamb,
				initialx=initial_vector.reshape(-1,1),
				initialscale=initial_scale,
				clipping=clipping_single)[0][:,0]

			print "Marta's xhat for irls = ", xhat_marta
			'''

			def custom_reg_function(a,y,lamb):
				return lp.least_squares(a,y)

			xhat_linvpy = lp.irls(
				matrix_a=a_gui,
				vector_y=y_gui,
				loss_function=lp.psi_huber,
				clipping=clipping_single,
				scale=initial_scale,
				lamb=lamb,
				initial_x=initial_vector,
				kind=test_kind,
				regularization=custom_reg_function)

			print "LinvPy's xhat for irls = ", xhat_linvpy
			print "real xhat = ", x
			print "=================================="

			# only tests this if noise is zero otherwise it fails all the time
			# because values are not EXACTLY the same
			if NOISE == 0 :
				np.testing.assert_array_almost_equal(x, xhat_linvpy, decimal=5)


if __name__ == '__main__':
	unittest.main()
