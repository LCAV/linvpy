import unittest
import numpy
from linear_inverse import regression
from tests import generate_random

TESTING_ITERATIONS = 300

class TestUM(unittest.TestCase):

    # preparing to test
    def setUp(self):
        """ Setting up for the test """
        #print "FooTest:setUp_:end"
     
    # ending the test
    def tearDown(self):
        """Cleaning up after the test"""
        #print "FooTest:tearDown_:begin"
        ## do something...
        #print "FooTest:tearDown_:end"
 
 	# Tests least_squares() on random inputs from size 2 to TESTING_ITERATIONS
    def test_least_squares(self):
    	for i in range(2,TESTING_ITERATIONS):
	        A,y = generate_random.generate_random(i)      
	        self.assertEquals(regression.least_squares(A,y).all(), numpy.linalg.lstsq(A,y)[0].all())
	        

if __name__ == '__main__':
    unittest.main()