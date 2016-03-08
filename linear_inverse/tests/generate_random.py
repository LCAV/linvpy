"""
    Generation of random (matrix, vector) tuples for test purpose.
"""

import numpy
from linear_inverse import regression

"""
Args:
    size (int): size of matrix and vector

Returns:
    tuple: a random tuple A,y
"""
def generate_random(size):
    if size == 1:
        return numpy.matrix([[numpy.random.rand()]]), [numpy.random.rand()]
    return numpy.matrix(numpy.random.rand(size,size-1)) , numpy.random.rand(size)