import numpy as np
import math

def least_squares(matrix_a, vector_y):
    '''
    Method computing the least squares solution
    :math:`\\hat x = {\\rm arg}\\min_x\\,\\lVert y - Ax \\rVert_2^2`.
    Basic algorithm to solve a linear inverse problem of the form y = Ax, where
    y (vector) and A (matrix) are known and x (vector) is unknown.

    :param matrix_a: (np.matrix) matrix A in y - Ax
    :param vector_y: (array) vector y in y - Ax

    :return array: vector x solution of least squares

    Example : compute the least squares solution of a system y = Ax

    .. code-block:: python

        import numpy as np
        import linvpy as lp

        A = np.matrix([[1,3],[3,4],[4,5]])
        y = [-6,1,-2]

        # Returns x_hat, the least squares solution of y = Ax
        lp.least_squares(A,y)

        # [ 3.86666667 -3.18666667]

    '''

    # Ensures np.matrix type
    matrix_a = np.matrix(matrix_a)

    # x = (A' A)^-1 A' y
    vector_x = np.dot(
        np.dot(
            np.linalg.inv(
                np.dot(
                    matrix_a.T, # A.T returns the transpose of A
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

    Example : compute the solution of a system y = Ax (knowing y,A) which is a
    tradeoff between the least squares solution and the minimization of x's
    L2-norm. The greater lambda, the smaller the norm of the given solution. 
    We take a matrix A which is ill-conditionned.

    .. code-block:: python

        import numpy as np
        import linvpy as lp

        A = np.matrix([[7142.80730214, 6050.32000196],
                       [6734.4239248, 5703.48709251],
                       [4663.22591408, 3949.23319264]])

        y = [0.83175086, 0.60012918, 0.89405644]

        # Returns x_hat, the tradeoff solution of y = Ax
        print lp.tikhonov_regularization(A, y, 50)

        # [8.25871731e-05   4.39467106e-05]

    '''

    if lambda_parameter < 0:
        raise ValueError('lambda_parameter must be zero or positive.')

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


def rho_huber(input, clipping=1.345):
    '''
    The regular huber loss function; the "rho" version.

    :math:`\\rho(x)=\\begin{cases}
    \\frac{1}{2}{x^2}& \\text{if |x| <=} clipping, \\\\
    clipping (|x| - \\dfrac{1}{2} clipping)& \\text{otherwise}.
    \\end{cases}`

    This function is quadratic for small inputs, and linear for large 
    inputs, with equal values and slopes of the different sections at the two
    points where |input|= clipping. The variable a often refers to the residuals, 
    that is to the difference between the observed and predicted values 
    a=y-f(x)

    :param input: (float) residual to be evaluated
    :param clipping: (optional)(float) clipping parameter 

    :return float: penalty incurred by the estimation

    Example : run huber loss on a vector

    .. code-block:: python

        import linvpy as lp

        x = [1,2,3,4,5,6,7,8,9]

        loss = [lp.rho_huber(e, 4) for e in x]

        # [0.5, 2.0, 4.5, 8.0, 12, 16, 20, 24, 28]
    '''
    # Casting input to float to avoid divisions rounding
    input = float(input)

    if clipping <= 0 :
        raise ValueError('clipping must be positive.')

    if (np.absolute(input) <= clipping):
        return math.pow(input, 2)/2
    else :
        return clipping * (np.subtract(np.absolute(input),clipping/2))


def psi_huber(input, clipping=1.345):
    '''
    Derivative of the Huber loss function; the "psi" version. Used in the weight 
    function of the M-estimator.

    :math:`\\psi(x)=\\begin{cases}
    x& \\text{if |x| <=} clipping, \\\\
    clipping * sign(x) & \\text{otherwise}.
    \\end{cases}`

    :param input: (float) residual to be evaluated
    :param clipping: (optional)(float) clipping parameter 

    :return float: penalty incurred by the estimation

    Example : run huber loss derivative on a vector

    .. code-block:: python

        import linvpy as lp

        x = [1,2,3,4,5,6,7,8,9]

        derivative = [lp.psi_huber(e, 4) for e in x]

        # [1, 2, 3, 4, 4, 4, 4, 4, 4]

    '''
    # Casting input to float to avoid divisions rounding
    input = float(input)

    if clipping <= 0 :
        raise ValueError('clipping must be positive.')

    if (np.absolute(input) >= clipping):
        return clipping * np.sign(input)
    else :
        return input

def rho_bisquare(input, clipping=4.685):
    '''
    The regular bisquare loss (or Tukey's loss), "rho" version.

    :math:`\\rho(x)=\\begin{cases}
    (c^2 / 6)(1-(1-(x/c)^2)^3)& \\text{if |x|} \\leq 0, \\\\
    c^2 / 6& \\text{if |x| > 0}.
    \\end{cases}`

    :param input: (float) residual to be evaluated
    :param clipping: (optional)(float) clipping parameter

    :return float: result of bisquare function

    Example : run huber loss on a vector

    .. code-block:: python

        import linvpy as lp

        x = [1,2,3,4,5,6,7,8,9]

        result = [lp.rho_bisquare(e, 4) for e in x]

        # [0.46940104166666663, 1.5416666666666665, 2.443359375, 2.6666666666666665, 2.6666666666666665, 2.6666666666666665, 2.6666666666666665, 2.6666666666666665, 2.6666666666666665]
    '''
    # Casting input to float to avoid divisions rounding
    input = float(input)

    if clipping <= 0 :
        raise ValueError('clipping must be positive.')

    if (np.absolute(input) <= clipping):
        return (
            (clipping**2.0)/6.0)*(
                1-(
                    (1-(
                        input/clipping)**2)
                    **3)
                )
    else :
        return (clipping**2)/6.0


def psi_bisquare(input, clipping=4.685):
    '''
    The derivative of bisquare loss (or Tukey's loss), "psi" version.

    :math:`\\psi(x)=\\begin{cases}
    x((1-(x/c)^2)^2)& \\text{if |x|} \\leq 0, \\\\
    0& \\text{if |x| > 0}.
    \\end{cases}`

    :param input: (float) residual to be evaluated
    :param clipping: (optional)(float) clipping parameter

    :return float: result of bisquare function

    Example : run huber loss on a vector

    .. code-block:: python

        import linvpy as lp

        x = [1,2,3,4,5,6,7,8,9]

        result = [lp.psi_bisquare(e, 4) for e in x]

        # [0.87890625, 1.125, 0.57421875, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    '''
    # Casting input to float to avoid divisions rounding
    input = float(input)

    if clipping <= 0 :
        raise ValueError('clipping must be positive.')

    if (np.absolute(input) <= clipping):
        return input*((1-(input/clipping)**2)**2)
    else :
        return 0.0

def rho_cauchy(input, clipping=2.3849):
    '''
    Cauchy loss function; the "rho" version.

    :math:`\\rho(x)=(c^2/2)log(1+(x/c)^2)`

    :param input: (float) residual to be evaluated
    :param clipping: (optional)(float) clipping parameter 

    :return float: result of the cauchy function

    Example : run huber loss on a vector

    .. code-block:: python

        import linvpy as lp

        x = [1,2,3,4,5,6,7,8,9]

        result = [lp.rho_cauchy(e, 4) for e in x]

        # [0.4849969745314787, 1.7851484105136781, 3.5702968210273562, 5.545177444479562, 7.527866755716213, 9.42923997073317, 11.214388381246847, 12.875503299472802, 14.416978050108813]
    '''
  # Casting input to float to avoid divisions rounding
    input = float(input)

    if clipping <= 0 :
        raise ValueError('clipping must be positive.')

    return (
        (clipping**2)/2
        )*math.log(
        1+(
            input/clipping)**2)

def psi_cauchy(input, clipping=2.3849):
    '''
    Derivative of Cauchy loss function; the "psi" version.

    :math:`\\psi(x)=\\frac{x}{1+(x/c)^2}`

    :param input: (float) residual to be evaluated
    :param clipping: (optional)(float) clipping parameter 

    :return float: result of the cauchy's derivative function

    Example : run huber loss on a vector

    .. code-block:: python

        import linvpy as lp

        x = [1,2,3,4,5,6,7,8,9]

        result = [lp.psi_cauchy(e, 4) for e in x]

        # [0.9411764705882353, 1.6, 1.92, 2.0, 1.951219512195122, 1.8461538461538463, 1.7230769230769232, 1.6, 1.4845360824742269]
    '''

    # Casting input to float to avoid divisions rounding
    input = float(input)

    if clipping <= 0 :
        raise ValueError('clipping must be positive.')

    return input/(
        1+(
            input/clipping
            )**2)

def rho_optimal(input, clipping=3.270):
    '''
    The so-called optimal 'rho' function is given by
    :math:`\\rho(x)=\\begin{cases}
    1.38(x/c)^2 & \\text{if |x/c|} \\leq 2/3, \\\\
    0.55 - 2.69(x/c)^2 + 10.76(x/c)^4 - 11.66(x/c)^6 + 4.04(x/c)^8 & \\text{if 2/3 }<|x/c| leq 1, \\\\
    1 &  \\text{if |x/c| > 1}.
    \\end{cases}`

    :param input: (float) residual to be evaluated
    :param clipping: (optional)(float) clipping parameter 

    :return float: result of the optimal function
    '''

    # TODO: ignore this (c = 3c). Implement the formula as it is
    # To get the exact formula, it is necessary to use 3*c instead of c.
    clipping = 3*clipping

    # Casting input to float to avoid divisions rounding
    input = float(input)

    if clipping <= 0 :
        raise ValueError('clipping must be positive.')
    # TODO: the implementation does not agrees with the formula, right?
    if abs(input/clipping) <= 2.0 :
        return 1.38 * (input/clipping)**2
    elif abs(input/clipping) <= 1.0 :
        return 0.55 - (2.69 * (input/clipping)**2) + (
            10.76 * (input/clipping)**4) - (
            11.66 * (input/clipping)**6) + (
            4.04 * (input/clipping)**8)
    elif abs(input/clipping) > 1 :
        return 1.0

def psi_optimal(input, clipping=3.270):
    '''
    The derivative of the optimal 'rho' function is given by
    :math:`\\rho(x)=\\begin{cases}
    2 1.38 x / c^2 & \\text{if |x/c|} \\leq 2/3, \\\\
    0.55 - 2 2.69x / c^2 + 4 10.76x^3 / c^4 - 6 11.66x^5/ c^6 + 8 4.04x^7 /c^8 & \\text{if 2/3 }<|x/c| leq 1, \\\\
    0 &  \\text{if |x/c| > 1}.
    \\end{cases}`

    :param input: (float) residual to be evaluated
    :param clipping: (optional)(float) clipping parameter 

    :return float: result of the optimal function
    '''

    # To get the exact formula, it is necessary to use 3*c instead of c.
    clipping = 3*clipping

    # Casting input to float to avoid divisions rounding
    input = float(input)

    if clipping <= 0 :
        raise ValueError('clipping must be positive.')


def weights(input, loss_function, clipping=None):
    '''
    Returns an array of :

    :math:`\\begin{cases}
    \\frac{loss\\_function(x_i)}{x_i}& \\text{if } x_i \\neq 0, \\\\
    0& \\text{otherwise}.
    \\end{cases}`

    Weights function designed to be used with loss functions like rho_huber, 
    psi_huber, rho_cauchy... Note that the loss_function passed in argument must
    support two inputs.

    :param input: (array or float) vector or float to be processed, x_i's
    :param loss_function: (loss_function) f(x) in f(x)/x.
    :param clipping: (optional) clipping parameter of the huber loss function.

    :return array or float: element-wise result of f(x)/x if x!=0, 0 otherwise

    Example : run the weight function with the psi_huber with default
    clipping or with another function like rho_cauchy and another clipping.

    .. code-block:: python

        import linvpy as lp

        x = [1,2,3,4,5]

        # psi_huber, default clipping
        lp.weights(x, lp.psi_huber)

        # [1.0, 0.67249999999999999, 0.44833333333333331, 0.33624999999999999, 0.26900000000000002]

        # rho_cauchy, clipping=2.5
        lp.weights(x, lp.rho_cauchy, 2.5)

        # [0.46381251599460444, 0.7729628778689174, 0.9291646242761568, 0.9920004256749526, 1.0058986952713127]

    '''

    # kwargs = keyword arguments : if clipping is not specified, kwargs=None
    # and we use the default loss function's clipping, otherwise we use the one
    # passed in weights() with **kwargs
    kwargs = {}
    if clipping != None:
        kwargs['clipping'] = clipping

    if isinstance(input, (int, float)):
        if (input == 0) :
            return 0.0
        return loss_function(input, **kwargs)/float(input)
    else :
        # Ensures the input is an array and not a matrix. 
        # Turns [[a b c]] into [a b c].

        input = np.squeeze(
                    np.asarray(
                        input
                        )
                    ).flatten()

        output = [0 if (i == 0) else 0.5*loss_function(i, **kwargs)/float(i) for i in input]
        return np.array(output)



# scale = sigma that divides; if sigma if given in parameter => preliminary scale
# lmb = lambda for tikhonov => if lambda is given : regularized m-estimator
# if lamb and scale are given : regularized m-estimator with preliminary scale
def irls(matrix_a, vector_y, loss_function, clipping=None, scale=None, lamb=0, 
    tolerance=1e-5, max_iterations=100):
    '''
    The method of iteratively reweighted least squares (IRLS) is used to solve
    certain optimization problems with objective functions of the form:

    :math:`\underset{ \\boldsymbol x } {\operatorname{arg\,min}} \sum_{i=1}^n | y_i - f_i (\\boldsymbol x)|^p`

    by an iterative method in which each step involves solving a weighted least 
    squares problem of the form:

    :math:`\\boldsymbol x^{(t+1)} = \underset{\\boldsymbol x} {\operatorname{arg\,min}} \sum_{i=1}^n w_i (\\boldsymbol x^{(t)})\\big| y_i - f_i (\\boldsymbol x) \\big|^2.`

    IRLS is used to find the maximum likelihood estimates of a generalized 
    linear model, and in robust regression to find an M-estimator, as a way of 
    mitigating the influence of outliers in an otherwise normally-distributed 
    data set. For example, by minimizing the least absolute error rather than 
    the least square error.

    :param matrix_a: (np.matrix) matrix A in y - Ax
    :param vector_y: (array) vector y in y - Ax
    :param loss_function: the loss function to be used in the M estimator
    :param clipping: clipping parameter for the loss function

    :return array: vector of x solution of IRLS

    '''

    # If a scale parameter is given, m-estimator runs with preliminary scale.
    # This checks that scale is int or float and nonzero.
    # If no scale is given, scale = 1.0
    if (scale != None) :
        if (isinstance(scale, float) or isinstance(scale, int)) == False :
            raise ValueError('scale must be a scalar.')
        elif scale == 0 :
            raise ValueError('scale must be nonzero.')
    else : scale = 1.0

    print "scale = ", scale

    # kwargs = keyword arguments : if clipping is not specified, kwargs=None
    # and we use the default loss function's clipping, otherwise we use the one
    # passed in weights() with **kwargs
    kwargs = {}
    if clipping != None:
        kwargs['clipping'] = clipping

    # Ensures numpy types
    matrix_a = np.matrix(matrix_a)
    vector_y = np.array(vector_y).flatten()

    # Generates a ones vector_x with length = matrix_a.columns
    vector_x = np.ones(matrix_a.shape[1])

    # Residuals = y - Ax, difference between measured values and model
    residuals = vector_y - np.dot(matrix_a, vector_x)
    residuals = residuals / scale

    for i in range(1,max_iterations):
        
        # weights(y-Ax, loss_function, clipping)
        weights_vector = weights(residuals, loss_function, **kwargs)

        # Makes a diagonal matrix with values of w(y-Ax)
        # np.squeeze(np.asarray()) is there to flatten the matrix into a vector
        weights_matrix = np.diag(
            np.squeeze(
                np.asarray(weights_vector)
                )
            )

        # Square root of the weights matrix, sqwm = W^1/2
        sqwm = np.sqrt(weights_matrix)

        # y_weighted = diagonal of W^1/2 y
        y_weighted = np.diagonal(sqwm * vector_y)

        # A_weighted = W^1/2 A
        a_weighted = sqwm * matrix_a

        # vector_x_new is there to keep the previous value to compare
        vector_x_new = tikhonov_regularization(a_weighted, y_weighted, lamb)

        # Normalized distance between previous and current iteration
        xdis = np.linalg.norm(vector_x - vector_x_new)

        # New residuals
        residuals = vector_y - np.dot(matrix_a, vector_x_new)

        # Divided by the specified optional scale, otherwise scale = 1
        residuals = residuals / scale
        vector_x = vector_x_new

        # if the difference between iteration n and iteration n+1 is smaller 
        # than tolerance, return vector_x
        if (xdis < tolerance):
            print('CONVERGED !')
            return vector_x

    print('DID NOT CONVERGE !')
    return vector_x


















#===============================================================================
#===================================TESTING AREA================================
#===============================================================================

