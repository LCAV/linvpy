#import unittest
import numpy as np
#from linvpy import regression
#import generate_random
import scipy

TESTING_ITERATIONS = 100
# For a matrix to be ill-conditioned, its condition number must be equal to or
# greather than ILL_CONDITION_CRITERIA
ILL_CONDITION_CRITERIA = 1000 

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
 
	# Tests least_squares() on random inputs from size 1 to TESTING_ITERATIONS
	def test_least_squares(self):
		for i in range(1,TESTING_ITERATIONS):
			A,y = generate_random.generate_random(i)      
			self.assertEquals(
				regression.least_squares(A,y).all(), 
				np.linalg.lstsq(A,y)[0].all()
				)

	
	# Tests least_squares_gradient() on inputs from size 1 to TESTING_ITERATIONS
	def test_least_squares_gradient(self):
		for i in range(1,TESTING_ITERATIONS):
			A,y = generate_random.generate_random(i)      
			self.assertAlmostEquals(
				regression.least_squares_gradient(A,y).all(), 
				np.linalg.lstsq(A,y)[0].all()
				)
	

	# Tests Tikhonov regularization against the native Scipy function
	def test_tikhonov(self):
		for i in range(2,TESTING_ITERATIONS):
			# Generates random lambda
			LAMBDA = np.random.rand(1)
			A,y = generate_random.generate_random_ill_conditioned(i)
			self.assertEquals(
				regression.tikhonov_regularization(A,y,LAMBDA).all(), 
				scipy.sparse.linalg.lsmr(A,y,LAMBDA)[0].all()
				)

	# Tests the ill-conditoned matrix generator
	# Checks that the condition number is greather than ILL_CONDITION_CRITERIA
	def test_ill_conditioned_matrix(self):
		for i in range(3,TESTING_ITERATIONS):
			self.assertTrue(
				np.linalg.cond(
					generate_random.generate_random_ill_conditioned(i)[0]
					) > ILL_CONDITION_CRITERIA
				)

if __name__ == '__main__':
	unittest.main()

