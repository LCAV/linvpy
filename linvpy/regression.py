'''
    Linear regression functions.
'''

import numpy as np
import math
import scipy
from scipy import special
from tests import generate_random

def least_squares(matrix_a, vector_y):
    '''
    Method computing the least squares solution
    :math:`\\hat x = {\\rm arg}\\min_x\\,\\lVert y - Ax \\rVert_2^2`.
    Basic algorithm to solve a linear inverse problem of the form y = Ax, where
    y (vector) and A (matrix) are known and x (vector) is unknown.

    :param matrix_a: (np.matrix) matrix A in y - Ax
    :param vector_y: (array) vector y in y - Ax

    :return array: vector x solution of least squares
    '''

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
    '''
    Method computing the least squares solution
    :math:`\\hat x = {\\rm arg}\\min_x\\,\\lVert y - Ax \\rVert_2^2,` using the 
    gradient descent algorithm.

    :param matrix_a: (np.matrix) matrix A in y - Ax
    :param vector_y: (array) vector y in y - Ax
    :param max_iterations: (optional)(int) number of iterations before stopping
     if the algorithm doesn't converge
    :param tolerance: (optional)(float) tolerance of the convergence criteria

    :return array: vector x solution of least squares

    :raises ValueError: raises an exception if max_iterations < 0
    :raises ValueError: raises an exception if tolerance < 0
    '''

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
    '''
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
    method). 

    The solution is given by :math:`\\hat{x} = (A^{T}A+ \\lambda^{2} I)^{-1}A^{T}\\mathbf{y}`, where I is the identity matrix.

    Raises a ValueError if lambda < 0.

    :param matrix_a: (np.matrix) matrix A in y - Ax
    :param vector_y: (array) vector y in y - Ax
    :param lambda: (int) lambda parameter to regulate the tradeoff

    :return array: vector_x solution of Tikhonov regularization

    :raises ValueError: raises an exception if lambda_parameter < 0
    '''

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


def huber_loss(input, delta=1.5):
    '''
    The Huber loss function describes the penalty incurred by an estimation 
    procedure f.

    :math:`L_\\delta (a) = \\frac{1}{2}{a^2} \\text{ for } |a| \\leq \\delta, \\delta (|a| - \\dfrac{1}{2} \\delta)  \\text{ otherwise }`

    This function is quadratic for small values of a, and linear for large 
    values, with equal values and slopes of the different sections at the two
    points where |a|= delta. The variable a often refers to the residuals, 
    that is to the difference between the observed and predicted values 
    a=y-f(x)

    :param input: (scalar) residual to be evaluated
    :param delta: (optional)(float) trigger parameter 

    :return float: penalty incurred by the estimation
    '''

    if delta <= 0 :
        raise ValueError("delta must be positive.")

    if (np.absolute(input) <= delta):
        return math.pow(input, 2)/2
    else :
        return delta * (np.subtract(np.absolute(input),delta/2))


def phi(input, delta=1.5):
    '''
    Phi(x), the derivative of the Huber loss function. Used in the weight 
    function of the M-estimator.

    :param input: (scalar) residual to be evaluated
    :param delta: (optional)(float) trigger parameter 

    :return float: penalty incurred by the estimation
    '''

    if delta <= 0 :
        raise ValueError("delta must be positive.")

    if (np.absolute(input) >= delta):
        return delta * np.sign(input)
    else :
        return input


def weight_function(input, function=phi, delta=1.5):
    '''
    Returns an array of [function(x_i)/x_i where x_i != 0, 0 otherwise.]
    By default the function is phi(x), the derivative of the Huber loss 
    function.

    :param input: (array or scalar) vector or scalar to be processed
    :param function: (optional)(function) f(x) in f(x)/x. phi(x) by default.
    :param delta: (optional) trigger parameter of the huber loss function.

    :return array or float: element-wise result of f(x)/x if x!=0, 0 otherwise
    '''

    # If the input is a list, the evaluation is run on all values and a list
    # is returned. If it's a scalar, a scalar is returned.
    if isinstance(input, (int, float)):
        return function(input, delta)
    else :
        # Ensures the input is an array and not a matrix. 
        # Turns [[a b c]] into [a b c].
        input = np.squeeze(
                    np.asarray(
                        input
                        )
                    )
        return [0 if (i == 0) else function(i, delta)/i for i in input]


def iteratively_reweighted_least_squares(matrix_a, vector_y):
    '''
    The method of iteratively reweighted least squares (IRLS) is used to solve
    certain optimization problems with objective functions of the form:

    :math:`\underset{ \\boldsymbol\\beta } {\operatorname{arg\,min}} \sum_{i=1}^n | y_i - f_i (\\boldsymbol\\beta)|^p`

    by an iterative method in which each step involves solving a weighted least 
    squares problem of the form:

    :math:`\\boldsymbol\\beta^{(t+1)} = \underset{\\boldsymbol\\beta} {\operatorname{arg\,min}} \sum_{i=1}^n w_i (\\boldsymbol\\beta^{(t)}) \\big| y_i - f_i (\\boldsymbol\\beta) \\big|^2.`

    IRLS is used to find the maximum likelihood estimates of a generalized 
    linear model, and in robust regression to find an M-estimator, as a way of 
    mitigating the influence of outliers in an otherwise normally-distributed 
    data set. For example, by minimizing the least absolute error rather than 
    the least square error.
    

    :param matrix_a: (np.matrix) matrix A in y - Ax
    :param vector_y: (array) vector y in y - Ax

    :return array: vector x solution of IRLS
    '''

    # Tolerance to estimate that the algorithm has converged
    TOLERANCE = 1e-5
    MAX_ITERATIONS = 100

    # Ensures numpy types
    matrix_a = np.matrix(matrix_a)
    vector_y = np.array(vector_y)

    # Generates a ones vector_x with length = matrix_a.columns
    vector_x = np.ones(matrix_a.shape[1])

    for x in xrange(1,MAX_ITERATIONS):
                
        # Makes a diagonal matrix with values of w(y-Ax)
        # f(x) on a numpy array applies the function to each element
        # np.squeeze(np.asarray()) is there to flatten the matrix into a vector
        weights_matrix = np.diag(
            np.squeeze(
                np.asarray(
                    # w(x) = phi(x)/x = huber_loss(x)/x = huber_loss(y-Ax)/(y-Ax)
                    weight_function(
                        np.subtract(vector_y,
                            np.dot(matrix_a,
                                vector_x
                                )
                            )
                        )
                    )
                )
            )


        # y_LS = W^1/2 y
        vector_y_LS = np.dot(
            np.sqrt(weights_matrix),
            vector_y
            )

        # A_LS = W^1/2 A
        matrix_a_LS = np.dot(
            np.sqrt(weights_matrix),
            matrix_a
            )

        # vector_x_storage is there to store the previous value to compare
        vector_x_storage = np.copy(vector_x)
        vector_x = least_squares(matrix_a_LS, vector_y_LS)

        '''
        print "weight matrix = ", weights_matrix
        print "vector_x_storage = ", vector_x_storage
        print "vector_x = ", vector_x
        print "matrix_a_LS = ", matrix_a_LS
        print "vector_y_LS = ", vector_y_LS
        '''

        # if the difference between iteration n and iteration n+1 is smaller 
        # than TOLERANCE, return vector_x
        if (np.linalg.norm(
            np.subtract(
                vector_x_storage, 
                vector_x
                )
            ) < TOLERANCE):
            print("CONVERGED !")
            return vector_x

    print("DID NOT CONVERGE !")
    return vector_x





#===============================================================================
#===================================TESTING AREA================================
#===============================================================================



# Dummy tests 2

A = np.matrix([[1,3],[3,4],[4,5]])
y = np.array([-6,1,-2])

def lol(a,b):
    return b*a

#print weight_function(-1000,huber_loss,332)

#print weight_function([4,0,2,0], phi)
#print phi(0)

#print huber_loss(-1,0.1)

#print phi([1,2,2])

#print "ITERATIVELY REWEIGHTED = ", iteratively_reweighted_least_squares(A,y)

#print scipy.special.bdtr(-1,10,0.3)
#print scipy.special.huber(1)

#print scipy.special.huber()

'''
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
'''

# Dummy tests 1
'''
A = np.matrix([[2,3],[3,4],[4,5]])
y = [1,2,3]
print "MY SOLUTION = ", least_squares(A,y)
# [0] is to take the first returned element of the lstsq function
print "np'S SOLUTION = " , np.linalg.lstsq(A,y)[0]
'''