import unittest
import numpy as np
import linvpy as lp
import generate_random
from scipy.sparse.linalg import lsmr
import matplotlib.pyplot as plt
import optimal as opt
import mestimator_marta as marta
import random

TESTING_ITERATIONS = 100
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
 
	# Tests least_squares() on random inputs from size 1 to TESTING_ITERATIONS
	def test_least_squares(self):
		for i in range(1,TESTING_ITERATIONS):
			A,y = generate_random.generate_random(i,i)      
			self.assertEquals(
				lp.least_squares(A,y).all(), 
				np.linalg.lstsq(A,y)[0].all()
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
	

	# Tests Tikhonov regularization against the native Scipy function
	def test_tikhonov(self):
		for i in range(2,TESTING_ITERATIONS):
			# Generates random lambda
			LAMBDA = np.random.rand(1)
			A,y = generate_random.generate_random_ill_conditioned(i)
			self.assertEquals(
				lp.tikhonov_regularization(A,y,LAMBDA).all(), 
				lsmr(A,y,LAMBDA)[0].all()
				)

	# Tests LinvPy's m-estimator against Marta's version
	def test_mestimator(self):
		for i in range(2, TESTING_ITERATIONS):
			A,y = generate_random.generate_random(i,i)
			y_marta = y.reshape(i,1)

			# random float clipping between 0 and 10
			clipping_test = random.uniform(0.0, 10.0)

			xhat_linvpy = lp.irls(
				A,
				y,
				lp.psi_huber,
				clipping=clipping_test)

			xhat_marta = marta.mestimator(
				y_marta,
				A,
				'huber',
				clipping=clipping_test)[0]

			#print "LinvPy's xhat = ", xhat_linvpy
			#print "Marta's xhat = ", xhat_marta

			self.assertEquals(xhat_linvpy.all(), xhat_marta.all())



# Plots loss functions
def plot_loss_functions():
	plt.plot([lp.rho_huber(i) for i in range(-PLOT_INTERVAL,PLOT_INTERVAL)], label="rho_huber")
	plt.plot([lp.psi_huber(i) for i in range(-PLOT_INTERVAL,PLOT_INTERVAL)], label="psi_huber")
	plt.plot([lp.rho_bisquare(i) for i in range(-PLOT_INTERVAL,PLOT_INTERVAL)], label="rho_bisquare")
	plt.plot([lp.psi_bisquare(i) for i in range(-PLOT_INTERVAL,PLOT_INTERVAL)], label="psi_bisquare")
	plt.plot([lp.rho_cauchy(i) for i in range(-PLOT_INTERVAL,PLOT_INTERVAL)], label="rho_cauchy")
	plt.plot([lp.psi_cauchy(i) for i in range(-PLOT_INTERVAL,PLOT_INTERVAL)], label="psi_cauchy")

	# Puts a legend box above the plots
	plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
	           ncol=2, mode="expand", borderaxespad=0.)

	# Displays the plots
	plt.show()

# Uncomment the following line to display plots :
#plot_loss_functions()

if __name__ == '__main__':
	unittest.main()
