import linvpy as lp
import mestimator_marta as marta
import generate_random as gen
import numpy as np
import copy
import random
import unittest

TESTING_ITERATIONS = 20

class TestUM(unittest.TestCase):

	def test_mestimator(self):

		for i in range(2, TESTING_ITERATIONS):

			A,y = gen.generate_random(i,i)
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

			print "Marta's xhat for m-estimator = ", xhat_marta
			print "LinvPy's xhat for m-estimator = ", xhat_linvpy

			self.assertEquals(xhat_linvpy.all(), xhat_marta.all())

if __name__ == '__main__':
	unittest.main()