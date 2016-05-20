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
		for i in range(3, TESTING_ITERATIONS):

			columns = random.randint(2,i)

			A,x,y = generate_random.gen_noise(i,columns,0.5)

			y_gui = copy.deepcopy(y)
			a_gui = copy.deepcopy(A)

			y_marta = copy.deepcopy(y.reshape(-1,1))
			a_marta = copy.deepcopy(A)

			# random float clipping between 0 and 10
			clipping_tau = (random.uniform(0.1, 4.0), random.uniform(0.1, 4.0))

			# a static initial vector x to avoid randomness in test (works only
			# with matrix,vector of size 2 !)
			#initial_vector = np.array([-0.56076046, -2.96528342]).reshape(-1,1)

			# a dynamic initial vector that has the right size
			initial_vector = generate_random.gen_noise(i,columns,0)[1]
			initial_residuals = y - np.dot(A, initial_vector)

			initial_scale = np.median(np.abs(initial_residuals))/0.6745

			test_kind='tau'

			xhat_marta = inv.irls(
				y=y_marta,
				a=a_marta,
				kind=test_kind,
				lossfunction='optimal',
				regularization='none',
				lmbd=0,
				initialx=initial_vector.reshape(-1,1),
				initialscale=initial_scale.reshape(-1,1),
				clipping=clipping_tau)[0][:,0].reshape(-1)

			print "Marta's xhat for tau irls = ", xhat_marta

			xhat_linvpy = lp.irls(
				matrix_a=a_gui,
				vector_y=y_gui,
				loss_function=lp.rho_optimal,
				clipping=clipping_tau,
				scale=initial_scale,
				lamb=0,
				initial_x=initial_vector,
				kind=test_kind)

			print "LinvPy's xhat for tau irls = ", xhat_linvpy
			print "real xhat = ", x
			print "=================================="

			#np.testing.assert_array_almost_equal(xhat_marta, xhat_linvpy, decimal=5)

			self.assertEquals(xhat_linvpy.all(), xhat_marta.all())


if __name__ == '__main__':
	unittest.main()
