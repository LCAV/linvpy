"""
    Linear regression functions. Uses types from Numpy package.
"""

import numpy


"""
Method computing the least squares solution min_x ||y - Ax||^2 using the gradient descent algorithm.

Args:
    matrix_a (numpy.matrix): matrix A in y - Ax
    vector_y (array): vector y in y - Ax

Returns:
    array: vector x solution of least squares
"""
def least_squares(matrix_a, vector_y):

    # Ensures numpy.matrix type
    matrix_a = numpy.matrix(matrix_a)

    # x = (A'A)^-1 A' y
    vector_x = numpy.dot(numpy.dot(numpy.linalg.inv(numpy.dot(matrix_a.transpose(), matrix_a)),matrix_a.transpose()),vector_y)

    # Flattens result into an array
    vector_x = numpy.squeeze(numpy.asarray(vector_x))

    return vector_x


# Dummy tests
"""
A = numpy.matrix([[1, 2],[3, 4],[4,5]])
y = [1,2,4]
print "MY SOLUTION = ", least_squares(A,y)
# [0] is to take the first returned element of the lstsq function
print "NUMPY'S SOLUTION = " , numpy.linalg.lstsq(A,y)[0]
"""