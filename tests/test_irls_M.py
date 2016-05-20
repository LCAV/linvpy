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

			columns = random.randint(2,i)

			A,x,y = generate_random.gen_noise(i,columns,0.5)

			y_gui = copy.deepcopy(y)
			a_gui = copy.deepcopy(A)

			y_marta = copy.deepcopy(y.reshape(-1,1))
			a_marta = copy.deepcopy(A)

			# random float clipping between 0 and 10
			clipping_single = random.uniform(0.1, 4.0)

			# a static initial vector x to avoid randomness in test
			#initial_vector = np.array([-0.56076046, -2.96528342]).reshape(-1,1)

			initial_vector = generate_random.gen_noise(i,columns,0)[1]

			initial_residuals = y - np.dot(A, initial_vector)

			initial_scale = np.median(np.abs(initial_residuals))/0.6745

			test_kind='M'

			# marta's irls returns a matrix with the same values repeated on
			# each line; so I take only the values of the first column
			xhat_marta = inv.irls(
				y=y_marta,
				a=a_marta,
				kind=test_kind,
				lossfunction='huber',
				regularization='none',
				lmbd=0,
				initialx=initial_vector.reshape(-1,1),
				initialscale=initial_scale,
				clipping=clipping_single)[0][:,0]

			print "Marta's xhat for irls = ", xhat_marta

			xhat_linvpy = lp.irls(
				matrix_a=a_gui,
				vector_y=y_gui,
				loss_function=lp.psi_huber,
				clipping=clipping_single,
				scale=initial_scale,
				lamb=0,
				initial_x=initial_vector,
				kind=test_kind)

			print "LinvPy's xhat for irls = ", xhat_linvpy
			print "real xhat = ", x
			print "=================================="

			# tests array equality to the 5th decimal
			#np.testing.assert_array_almost_equal(x, xhat_linvpy, decimal=5)

			self.assertEquals(xhat_linvpy.all(), x.all())


if __name__ == '__main__':
	unittest.main()
