"""
    Linear regression functions.
"""

import numpy as np
import math
from sklearn import linear_model
import scipy
from tests import generate_random

def least_squares(matrix_a, vector_y):
    """
    Method computing the least squares solution min_x ||y - Ax||^2.

    :param matrix_a: (np.matrix) matrix A in y - Ax
    :param vector_y: (array) vector y in y - Ax

    :return array: vector x solution of least squares
    """

    # Ensures np.matrix type
    matrix_a = np.matrix(matrix_a)

    # x = (A' A)^-1 A' y
    vector_x = np.dot(
        np.dot(
            np.linalg.inv(
                np.dot(
                    matrix_a.T, #A.T returns the transpose of A
                    matrix_a
                    )
            ),
            matrix_a.T
        ),
        vector_y
    )

    # Flattens result into an array
    vector_x = np.squeeze(np.asarray(vector_x))

    return vector_x


def least_squares_gradient(matrix_a, vector_y, max_iterations=100,
    tolerance=1e-6):
    """
    Method computing the least squares solution of the problem : min_x ||y -
    Ax||^2 using the gradient descent algorithm.

    :param matrix_a: (np.matrix) matrix A in y - Ax
    :param vector_y: (array) vector y in y - Ax
    :param max_iterations: (optional)(int) number of iterations before stopping
     if the algorithm doesn't converge
    :param tolerance: (optional)(float) tolerance of the convergence criteria

    :return array: vector x solution of least squares

    :raises ValueError: raises an exception if max_iterations < 0
    :raises ValueError: raises an exception if tolerance < 0
    """

    if max_iterations < 0 or tolerance < 0 :
        raise ValueError("max_iterations and tolerance must be zero or positive.")

    ALPHA = 0.01

    # ensures numpy.matrix type
    matrix_a = np.matrix(matrix_a)
    N = len(vector_y)

    # initialize beta : a vector of 0 whose length is equal to the number of columns of the matrix
    beta = np.zeros(matrix_a.shape[1])

    for i in range(1,max_iterations):

        # Compute the error : error = y - A * beta
        # numpy.squeeze(numpy.asarray(v)) is used to ensure the array type of v
        # numpy.dot(a,b) = dot product of a and b
        error = vector_y - np.squeeze(
            np.asarray(
                np.dot(matrix_a, beta)
                )
            )

        # Compute gradient : gradient = - (matrix_a_transposed * error) / N
        gradient = - np.divide(
            np.dot(matrix_a.transpose(), error
                ),
            N
            )
        
        # Flattens matrix to vector
        gradient = np.squeeze(np.asarray(gradient))

        # Update beta
        beta = np.subtract(beta, ALPHA * gradient)

        # Convergence criteria
        if np.dot(gradient.transpose(), gradient) < tolerance:
            break

    # Returns vector beta solution to the least squares problem
    return beta


def tikhonov_regularization(matrix_a, vector_y, lambda_parameter):
    """
    The standard approach to solve Ax=y (x is unknown) is ordinary least squares
    linear regression. However if no x satisfies the equation or more than one x
    does -- that is the solution is not unique -- the problem is said to be
    ill-posed. In such cases, ordinary least squares estimation leads to an
    overdetermined (over-fitted), or more often an underdetermined 
    (under-fitted) system of equations.

    The Tikhonov regularization is a tradeoff between the least squares 
    solution and the minimization of the L2-norm of the output x (L2-norm = 
    sum of squared values of the vector x). 
    
    The parameter lambda tells how close to the least squares solution the 
    output x will be; a large lambda will make x close to L2-norm(x)=0, while 
    a small lambda will approach the least squares solution (typically running 
    the function with lambda=0 will behave like the normal leat_squares() 
    method). The solution is given by x = (A'A + lambda^2 I)^-1 A'y, where I is
    the identity matrix and A' the transpose of A.

    Raises a ValueError if lambda < 0.

    :param matrix_a: (np.matrix) matrix A in y - Ax
    :param vector_y: (array) vector y in y - Ax
    :param lambda: (int) lambda parameter to regulate the tradeoff

    :return array: vector_x solution of Tikhonov regularization

    :raises ValueError: raises an exception if lambda_parameter < 0
    """

    if lambda_parameter < 0:
        raise ValueError("lambda_parameter must be zero or positive.")

    # Ensures np.matrix type
    matrix_a = np.matrix(matrix_a)

    # Generates an identity matrix of the same shape as A'A.
    # matrix_a.shape() returns a tuple (#row,#columns) so with [1] with take the
    # number of columns to build the identity because A'A yields a square
    # matrix of the same size as the number of columns of A and rows of A'.
    identity_matrix = np.identity(matrix_a.shape[1])

    # x = (A' A + lambda^2 I)^-1 A' y
    vector_x = np.dot(
        np.dot(
            np.linalg.inv(
                np.add(
                    np.dot(matrix_a.T, matrix_a), # A.T transpose of A
                    np.dot(math.pow(lambda_parameter,2), identity_matrix)
                ),
            ),
            matrix_a.T
        ),
        vector_y
    )

    # Flattens result into an array
    vector_x = np.squeeze(np.asarray(vector_x))

    return vector_x

# Does not work correctly for now. Avoid using this.
def ridge(A,y,lambda_parameter):
    clf = linear_model.Ridge(alpha=lambda_parameter)
    clf.fit(A, y)
    linear_model.Ridge(alpha=lambda_parameter, copy_X=True, fit_intercept=False,
                   max_iter=None, normalize=False, random_state=None, 
                   solver='auto', tol=.0000001)
    return clf.coef_



# Dummy tests 2
"""

A = np.matrix([[1,3],[3,4],[4,5],[1,2]])
y = [1,2,3,2]
A_lambda = -1.5

#print "LEAST SQUARES =", least_squares(y,A)
#print "LEAST SQUARES GRADIENT =", least_squares_gradient(A,y,100,1)
#print "TIKHONOV=", tikhonov_regularization(A,y,A_lambda)
B = np.matrix([[0, 1], [0, 0], [1, 1]])
yy = np.array([0,0.1,1])
this_lambda = 10

print "RIDGE =", ridge(B,yy,this_lambda)
print "SCIPY =", scipy.sparse.linalg.lsmr(B,yy,this_lambda)[0]


print "ill conditioned : ", generate_random.generate_random_ill_conditioned(5)[1]

print "SCIPY TEST : ", scipy.sparse.linalg.lsmr(
    generate_random.generate_random_ill_conditioned(5)[0],
    generate_random.generate_random_ill_conditioned(5)[1],
    this_lambda)[0]

A,y = generate_random.generate_random_ill_conditioned(5)
print "A=", A
print "y=", y

print "SCIPY TEST 2 : ", scipy.sparse.linalg.lsmr(A,y,this_lambda)[0]
print "tikhonov test 2 ", tikhonov_regularization(A,y, this_lambda)

#print "MY SOLUTION LS = ", least_squares(A,y)
# [0] is to take the first returned element of the lstsq function
#print "numpy'S SOLUTION = " , np.linalg.lstsq(B,yy)[0]
#print "WEB SOLUTION =", testing(y,A,this_lambda)
"""

# Dummy tests 1
"""
A = np.matrix([[2,3],[3,4],[4,5]])
y = [1,2,3]
print "MY SOLUTION = ", least_squares(A,y)
# [0] is to take the first returned element of the lstsq function
print "np'S SOLUTION = " , np.linalg.lstsq(A,y)[0]
"""