import unittest
import numpy as np
import linvpy as lp
import generate_random
from scipy.sparse.linalg import lsmr
import matplotlib.pyplot as plt
import optimal as opt
import mestimator_marta as marta
import random
import copy
import toolboxinverse as inv

TESTING_ITERATIONS = 10
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

			#print "Marta's xhat for m-estimator = ", xhat_marta
			#print "LinvPy's xhat for m-estimator = ", xhat_linvpy

			self.assertEquals(xhat_linvpy.all(), xhat_marta.all())

	# Tests LinvPy's basictau against Marta's version
	def test_basictau(self):
		for i in range(2, TESTING_ITERATIONS):
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

			# max number of iterations for irls
			max_iter = random.randint(1,100)

			# how many solutions do we keep
			n_best = random.randint(1,20)

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

			# asserts that xhat's are equals and shat's are equals
			self.assertEquals(xhat_lp.all(), xhat_marta.all())
			self.assertEquals(shat_lp.all(), shat_marta.all())



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
