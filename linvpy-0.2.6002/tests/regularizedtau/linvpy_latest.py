from __future__ import division  # take the division operator from future versions
import numpy as np
import math
from sklearn import linear_model
from regularizedtau import toolboxutilities_latest as util
import matplotlib.pyplot as plt


# import matlab.engine



def least_squares(matrix_a, vector_y):
    '''
    This function computes the estimate :math:`\\hat x` given by the least squares method
    :math:`\\mathbf{\\hat x} = {\\rm arg}\\min_x\\,\\lVert \\mathbf{y - Ax} \\rVert_2^2`.
    This is the simplest algorithm to solve a linear inverse problem of the form :math:`\\mathbf{y = Ax + n}`, where
    :math:`\\mathbf{y}` (vector) and :math:`\\mathbf{A}` (matrix) are known and :math:`\\mathbf{x}`  (vector)
     and :math:`\\mathbf{n}`  (vector) are unknown.

    :param matrix_a: (np.matrix) matrix :math:`\\mathbf{A}`
    :param vector_y: (array) vector :math:`\\mathbf{y}`

    :return vector_x: (array) estimate :math:`\\mathbf{\\hat x}` given by least squares

    Example : compute the least squares solution of a system :math:`\\mathbf{y = Ax}`

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
                    matrix_a.T,  # A.T returns the transpose of A
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


def tikhonov_regularization(matrix_a, vector_y, lambda_parameter=0):
    '''
    The standard approach to solve the problem :math:`\\mathbf{y = Ax + n}` explained above is to use the  ordinary
    least squares method. However if your matrix :math:`\\mathbf{A}` is a fat matrix (it has more columns than rows)
    or it has a large condition number, then you should use a regularization to your problem in order to get a
    meaningful estimation of :math:`\\mathbf{x}`.

    The Tikhonov regularization is a tradeoff between the least squares 
    solution and the minimization of the L2-norm of the output :math:`x` (L2-norm =
    sum of squared values of the vector :math:`x`),
    :math:`\\hat{\\mathbf{x}} = {\\rm arg}\\min_x\\,\\lVert \\mathbf{y - Ax} \\rVert_2^2 + \\lambda\\lVert \\mathbf{x} \\rVert_2^2`
    
    The parameter lambda tells how close to the least squares solution the
    output :math:`\\mathbf{x}` will be; a large lambda will make :math:`\\mathbf{x}` close to
    :math:`\\lVert\\mathbf{x}\\rVert_2^2 = 0`, while
    a small lambda will approach the least squares solution (Running
    the function with lambda=0 will behave like the ordinary least_squares()
    method). 

    The Tikhonov solution has an analytic solution and it is given
    by :math:`\\hat{\\mathbf{x}} = (\\mathbf{A^{T}A}+ \\lambda^{2} \\mathbf{I})^{-1}\\mathbf{A}^{T}\\mathbf{y}`,
    where :math:`\\mathbf{I}` is the identity matrix.

    Raises a ValueError if lambda < 0.

    :param matrix_a: (np.matrix) matrix A in :math:`\\mathbf{y = Ax + n}`
    :param vector_y: (array) vector y in :math:`\\mathbf{y = Ax + n}`
    :param lambda: (int) lambda non-negative parameter to regulate the trade off.

    :return vector_x: (array) Tikhonov estimate :math:`\\hat{\\mathbf{x}}`

    :raises ValueError: raises an exception if lambda_parameter < 0

    Example : compute the solution of a system :math:`\\mathbf{y = Ax}` (knowing :math:`\\mathbf{y, A}`) which is a
    trade off between the least squares solution and the minimization of x's
    L2-norm. The greater lambda, the smaller the norm of the given solution. 
    We take a matrix :math:`\\mathbf{A}` which is ill-conditionned.

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
    if lambda_parameter == 0:
        return np.linalg.lstsq(matrix_a, vector_y)[0].reshape(-1)

    # Ensures np.matrix type
    matrix_a = np.matrix(matrix_a)

    # Generates an identity matrix of the same shape as A'A.
    # matrix_a.shape() returns a tuple (#row,#columns) so with [1] with take the
    # number of columns to build the identity because A'A yields a square
    # matrix of the same size as the number of columns of A and rows of A'.
    identity_matrix = np.identity(matrix_a.shape[1])

    try:
        # x = (A' A + lambda I)^-1 A' y
        vector_x = np.dot(
            np.dot(
                np.linalg.inv(
                    np.add(
                        np.dot(matrix_a.T, matrix_a),  # A.T transpose of A
                        np.dot(lambda_parameter, identity_matrix)
                    ),
                ),
                matrix_a.T
            ),
            vector_y
        )

        # Flattens result into an array
        vector_x = np.squeeze(np.asarray(vector_x))
        return vector_x

    # catches Singular matrix error (or any other error), prints the trace
    except:
        print("Lambda parameter may be too small.")
        raise


def lasso_regularization(matrix_a, vector_y, lambda_parameter=0):
    """
    Lasso algorithm that solves min ||y - Ax||_2^2 + lambda ||x||_1
    :param matrix_a:
    :param vector_y:
    :param lambda_parameter:
    :return: estimated x
    """

    # convert regularization parameter (sklearn considers (1/2m factor))
    reg_parameter = lambda_parameter / (2 * len(vector_y))

    # initialize model
    clf = linear_model.Lasso(reg_parameter, fit_intercept=False, normalize=False)

    # fit it
    clf.fit(matrix_a, vector_y)

    # return estimate
    x = clf.coef_

    return x


def rho_huber(input, clipping=1.345):
    '''
    The regular huber loss function; the "rho" version.

    :math:`\\rho(x)=\\begin{cases}
    \\frac{1}{2}{x^2}& \\text{if |x| <=} clipping, \\\\
    clipping (|x| - \\dfrac{1}{2} clipping)& \\text{otherwise}.
    \\end{cases}`

    This function is quadratic for small inputs, and linear for large 
    inputs.

    :param input: (float) :math:`x`
    :param clipping: (optional)(float) clipping parameter. Default value is optimal for normalized distributions.

    :return float: :math:`\\rho(x)`

    Example : run huber loss on a vector

    .. code-block:: python

        import linvpy as lp

        x = [1,2,3,4,5,6,7,8,9]

        loss = [lp.rho_huber(e, 4) for e in x]

        # [0.5, 2.0, 4.5, 8.0, 12, 16, 20, 24, 28]
    '''
    # Casting input to float to avoid divisions rounding
    input = float(input)

    if clipping <= 0:
        raise ValueError('clipping must be positive.')

    if (np.absolute(input) <= clipping):
        return math.pow(input, 2) / 2.0
    else:
        return clipping * (np.subtract(np.absolute(input), clipping / 2.0))


def psi_huber(input, clipping=1.345):
    '''
    Derivative of the Huber loss function; the "psi" version. Used in the weight 
    function of the M-estimator.

    :math:`\\psi(x)=\\begin{cases}
    x& \\text{if |x| <=} clipping, \\\\
    clipping \\cdot sign(x) & \\text{otherwise}.
    \\end{cases}`

    :param input: (float) :math:`x`
    :param clipping: (optional)(float) clipping parameter. Default value is optimal for normalized distributions.

    :return float: :math:`\\psi(x)`

    Example : run psi_huber derivative on a vector

    .. code-block:: python

        import linvpy as lp

        x = [1,2,3,4,5,6,7,8,9]

        derivative = [lp.psi_huber(e, 4) for e in x]

        # [1, 2, 3, 4, 4, 4, 4, 4, 4]

    '''
    # Casting input to float to avoid divisions rounding
    input = float(input)

    if clipping <= 0:
        raise ValueError('clipping must be positive.')

    if (np.absolute(input) >= clipping):
        return clipping * np.sign(input)
    else:
        return input


def rho_bisquare(input, clipping=4.685):
    '''
    The regular bisquare loss (or Tukey's loss), "rho" version.

    :math:`\\rho(x)=\\begin{cases}
    (c^2 / 6)(1-(1-(x/c)^2)^3)& \\text{if |x|} \\leq 0, \\\\
    c^2 / 6& \\text{if |x| > 0}.
    \\end{cases}`

    :param input: (float) :math:`x`
    :param clipping: (optional)(float) clipping parameter. Default value is optimal for normalized distributions.

    :return: (float) result :math:`\\rho(x)` of bisquare function

    Example : run bisquare loss on a vector

    .. code-block:: python

        import linvpy as lp

        x = [1,2,3,4,5,6,7,8,9]

        result = [lp.rho_bisquare(e, 4) for e in x]

        # [0.46940104166666663, 1.5416666666666665, 2.443359375, 2.6666666666666665, 2.6666666666666665, 2.6666666666666665, 2.6666666666666665, 2.6666666666666665, 2.6666666666666665]
    '''
    # Casting input to float to avoid divisions rounding
    input = float(input)

    if clipping <= 0:
        raise ValueError('clipping must be positive.')

    if (np.absolute(input) <= clipping):
        return (
                   (clipping ** 2.0) / 6.0) * (
                   1 - (
                       (1 - (
                           input / clipping) ** 2)
                       ** 3)
               )
    else:
        return (clipping ** 2) / 6.0


def psi_bisquare(input, clipping=4.685):
    '''
    The derivative of bisquare loss (or Tukey's loss), "psi" version.

    :math:`\\psi(x)=\\begin{cases}
    x((1-(x/c)^2)^2)& \\text{if |x|} \\leq 0, \\\\
    0& \\text{if |x| > 0}.
    \\end{cases}`

    :param input: (float) :math:`x`
    :param clipping: (optional)(float) clipping parameter. Default value is optimal for normalized distributions.

    :return: (float) :math:`\\psi(x)`

    Example : run psi_bisquare on a vector

    .. code-block:: python

        import linvpy as lp

        x = [1,2,3,4,5,6,7,8,9]

        result = [lp.psi_bisquare(e, 4) for e in x]

        # [0.87890625, 1.125, 0.57421875, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    '''
    # Casting input to float to avoid divisions rounding
    input = float(input)

    if clipping <= 0:
        raise ValueError('clipping must be positive.')

    if (np.absolute(input) <= clipping):
        return input * ((1 - (input / clipping) ** 2) ** 2)
    else:
        return 0.0


def rho_cauchy(input, clipping=2.3849):
    '''
    Cauchy loss function; the "rho" version.

    :math:`\\rho(x)=(c^2/2)log(1+(x/c)^2)`

    :param input: (float) :math:`x`
    :param clipping: (optional)(float) clipping parameter. Default value is optimal for normalized distributions.

    :return float: :math:`\\rho(x)`

    Example : run Cauchy loss on a vector

    .. code-block:: python

        import linvpy as lp

        x = [1,2,3,4,5,6,7,8,9]

        result = [lp.rho_cauchy(e, 4) for e in x]

        # [0.4849969745314787, 1.7851484105136781, 3.5702968210273562, 5.545177444479562, 7.527866755716213, 9.42923997073317, 11.214388381246847, 12.875503299472802, 14.416978050108813]
    '''
    # Casting input to float to avoid divisions rounding
    input = float(input)

    if clipping <= 0:
        raise ValueError('clipping must be positive.')

    return (
               (clipping ** 2) / 2
           ) * math.log(
        1 + (
            input / clipping) ** 2)


def psi_cauchy(input, clipping=2.3849):
    '''
    Derivative of Cauchy loss function; the "psi" version.

    :math:`\\psi(x)=\\frac{x}{1+(x/c)^2}`

    :param input: (float) :math:`x`
    :param clipping: (optional)(float) clipping parameter. Default value is optimal for normalized distributions.

    :return float: result of the Cauchy's derivative function

    Example : run psi_cauchy on a vector

    .. code-block:: python

        import linvpy as lp

        x = [1,2,3,4,5,6,7,8,9]

        result = [lp.psi_cauchy(e, 4) for e in x]

        # [0.9411764705882353, 1.6, 1.92, 2.0, 1.951219512195122, 1.8461538461538463, 1.7230769230769232, 1.6, 1.4845360824742269]
    '''

    # Casting input to float to avoid divisions rounding
    input = float(input)

    if clipping <= 0:
        raise ValueError('clipping must be positive.')

    return input / (
        1 + (
            input / clipping
        ) ** 2)


def rho_optimal(input, clipping=3.270):
    '''
    The so-called optimal loss function is given by
    :math:`\\rho(x)=\\begin{cases}
    1.38(x/c)^2 & \\text{if |x/c|} \\leq 2/3, \\\\
    0.55 - 2.69(x/c)^2 + 10.76(x/c)^4 - 11.66(x/c)^6 + 4.04(x/c)^8 & \\text{if 2/3 }<|x/c| \\leq 1, \\\\
    1 &  \\text{if |x/c| > 1}.
    \\end{cases}`

    :param input: (float) :math:`x`
    :param clipping: (optional)(float) clipping parameter. Default value is optimal for normalized distributions.

    :return float: :math:`\\rho(x)`
    '''

    # Casting input to float to avoid divisions rounding
    input = float(input)

    if clipping <= 0:
        raise ValueError('clipping must be positive.')
    if abs(input / clipping) <= 2.0 / 3.0:
        return 1.38 * (input / clipping) ** 2
    elif abs(input / clipping) <= 1.0:
        return 0.55 - (2.69 * (input / clipping) ** 2) + (
            10.76 * (input / clipping) ** 4) - (
                   11.66 * (input / clipping) ** 6) + (
                   4.04 * (input / clipping) ** 8)
    elif abs(input / clipping) > 1:
        return 1.0


def psi_optimal(input, clipping=3.270):
    '''
    The derivative of the optimal 'rho' function is given by
    :math:`\\psi(x)=\\begin{cases}
    2\\cdot1.38 x / c^2 & \\text{if |x/c|} \\leq 2/3, \\\\
    2\\cdot2.69x / c^2 + 4\\cdot10.76x^3 / c^4 - 6\\cdot11.66x^5/ c^6 + 8\\cdot4.04x^7 /c^8 & \\text{if 2/3 }<|x/c| \\leq 1, \\\\
    0 &  \\text{if |x/c| > 1}.
    \\end{cases}`


    :param input: (float) :math:`x`
    :param clipping: (optional)(float) clipping parameter. Default value is optimal for normalized distributions.

    :return float: :math:`\\psi(x)`
    '''

    if clipping <= 0:
        raise ValueError('clipping must be positive.')

    if abs(input / clipping) <= 2.0 / 3.0:
        return 2 * 1.38 * (input / clipping ** 2)
    elif abs(input / clipping) <= 1.0:
        return (- 2 * 2.69 * (input / clipping ** 2)) + (
            4 * 10.76 * (input ** 3 / clipping ** 4)) - (
                   6 * 11.66 * (input ** 5 / clipping ** 6)) + (
                   8 * 4.04 * (input ** 7 / clipping ** 8))
    elif abs(input / clipping) > 1:
        return 0


def weights(input, loss_function, clipping=None, nmeasurements=None):
    '''
    Returns an array of :

    :math:`\\begin{cases}
    \\frac{loss\\_function(x_i)}{x_i}& \\text{if } r_i \\neq 0, \\\\
    1& \\text{otherwise}.
    \\end{cases}`

    Weights function designed to be used with loss functions like rho_huber, 
    psi_huber, rho_cauchy... Note that the loss_function passed in argument must
    support two inputs.

    :param input: (array or float) vector or float to be processed, r_i's
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
        if (input == 0):
            return 1.0
        # print 'ENTER 1'
        # return loss_function(input, **kwargs)/float(input)
        return util.scorefunction(input, 'huber', **kwargs) / float(input)

    # only used for the tau estimator
    elif nmeasurements != None:

        # print 'Marta ENTER 2, rhat = ', input

        z = util.scorefunction(input, 'tau', **kwargs)

        # print 'Marta z = ', z

        # for zero inputs, set weight to one
        w = np.ones(input.shape)

        # only for the non zero u elements
        i = np.nonzero(input)
        w[i] = z[i] / (2 * input[i] * nmeasurements)

        # print 'Marta w = ', w

        return w

    else:
        # Ensures the input is an array and not a matrix. 
        # Turns [[a b c]] into [a b c].

        # print 'ENTER 3!'

        input = np.squeeze(
            np.asarray(
                input
            )
        ).flatten()

        # output = [0 if (i == 0) else 0.5*loss_function(i, **kwargs)/float(i) for i in input]
        output = [1 if (i == 0) else 0.5 * util.scorefunction(i, 'huber', **kwargs) / float(i) for i in input]
        return np.array(output)


# scale = sigma that divides; if sigma if given in parameter => preliminary scale
# lmb = lambda for tikhonov => if lambda is given : regularized m-estimator
# if lamb and scale are given : regularized m-estimator with preliminary scale
def irls(
        matrix_a,
        vector_y,
        loss_function,
        clipping=None,
        scale=None,
        lamb=0,
        initial_x=None,
        regularization=tikhonov_regularization,
        kind=None,
        b=0.5,
        tolerance=1e-10,
        max_iterations=100
):
    '''
    The method of iteratively reweighted least squares (IRLS) minimizes iteratively the function:

    :math:`\\boldsymbol x^{(t+1)} =`
    :math:`{\\rm arg}\\min_x\\,`
    :math:`\\big|\\big| \\boldsymbol W (\\boldsymbol x^{(t)})(\\boldsymbol y - \\boldsymbol A \\boldsymbol x )\\big | \\big |_2^2.`

    The IRLS is used, among other things, to compute the M-estimate and the tau-estimate.

    :param matrix_a: (np.matrix) matrix A in y - Ax
    :param vector_y: (ndarray) vector y in y - Ax
    :param loss_function: the loss function to be used in the M estimator
    :param clipping: clipping parameter for the loss function

    :return array: vector x solution of IRLS

    '''

    # print 'Marta 0 A,y = ', matrix_a,vector_y

    # If a scale parameter is given, m-estimator runs with preliminary scale.
    # This checks that scale is int or float and nonzero.
    # If no scale is given, scale = 1.0
    if (scale == None) or (scale == 0):
        scale = 1.0

    # kwargs = keyword arguments : if clipping is not specified, kwargs=None
    # and we use the default loss function's clipping, otherwise we use the one
    # passed in weights() with **kwargs
    # If the tau option is used, the function needs a clipping as tuple,
    # otherwise only one clipping is given.
    kwargs = {}
    if clipping != None:
        kwargs['clipping'] = clipping

    # if an initial value for x is specified, use it, otherwise generate a
    # vector of ones
    if initial_x is not None:
        vector_x = initial_x
    else:
        # Generates a ones vector_x with length = matrix_a.columns
        vector_x = np.ones(matrix_a.shape[1])
        initial_x = np.ones(matrix_a.shape[1])

    # number of measurements and unknowns; by default None, if tau is used it
    # takes value m,n = matrix_a.shape
    m = None

    # Ensures numpy types
    matrix_a = np.matrix(matrix_a)
    vector_y = vector_y.reshape(-1, 1)

    # Residuals = y - Ax, difference between measured values and model
    residuals = vector_y - np.dot(matrix_a, initial_x).reshape(-1, 1)

    # print 'Marta residuals 1 = ', residuals

    for i in range(1, max_iterations):

        # if we are computing the tau estimator, we need to upgrade the estimation of the scale in each iteration
        if kind == 'tau':

            # print 'Marta TAU'

            m, n = matrix_a.shape

            residuals = np.asarray(residuals.reshape(-1)).flatten()

            # print 'Marta residuals 2 = ', residuals

            # scale = scale * (mean(loss_function(residuals/scale))/b)^1/2
            if (scale != 0):
                scale *= np.sqrt(
                    np.mean(
                        # array_loss(residuals / scale, loss_function, clipping[0])
                        util.rhofunction(residuals / scale, loss_function, clipping[0])
                    ) / b
                )

        # normalize residuals ((y - Ax)/ scale)
        if (scale != 0):
            rhat = np.array(residuals / scale).flatten()
        else:
            rhat = np.array(residuals).flatten()

        # print "Marta rhat = ", rhat
        #
        # print "Marta irls rhat = ", rhat
        # print 'Marta irls scale = ', scale
        # print 'Marta irls residuals = ', residuals

        # weights(y-Ax, loss_function, clipping)
        weights_vector = weights(rhat, loss_function, nmeasurements=m, **kwargs)

        # print "Marta weights vector 1 = ", weights_vector

        # Makes a diagonal matrix with values of w(y-Ax)
        # np.squeeze(np.asarray()) is there to flatten the matrix into a vector
        weights_matrix = np.diag(
            np.squeeze(
                np.asarray(weights_vector)
            )
        )

        # Square root of the weights matrix, sqwm = W^1/2
        # sqwm = np.sqrt(weights_vector.reshape(-1,1))
        sqwm = np.sqrt(weights_matrix)


        # print 'sqwm, A = ', sqwm, matrix_a

        # A_weighted = W^1/2 A
        a_weighted = np.dot(sqwm, matrix_a)

        # y_weighted = diagonal of W^1/2 y
        # sqwm = sqwm.reshape(-1)

        y_weighted = np.dot(sqwm, vector_y)

        # vector_x_new is there to keep the previous value to compare
        vector_x_new = regularization(a_weighted, y_weighted, lamb)

        # ensures flattened array type
        vector_x = np.asarray(vector_x).flatten()

        # print 'Marta vector x, vector x new = ', vector_x, vector_x_new

        # Normalized distance between previous and current iteration
        xdis = np.linalg.norm(vector_x - vector_x_new)

        # print 'Marta xdis =', xdis
        # print 'x irls ', vector_x_new

        # TODO : changed by Guillaume
        vector_x_new = np.array(vector_x_new).reshape(-1)

        # New residuals
        residuals = vector_y.reshape(-1) - np.dot(matrix_a, vector_x_new).reshape(-1)

        # Divided by the specified optional scale, otherwise scale = 1
        # residuals = np.array(residuals / scale).flatten()

        vector_x = vector_x_new

        # if the difference between iteration n and iteration n+1 is smaller 
        # than tolerance, return vector_x
        if (xdis < tolerance):
            return vector_x

    return vector_x


def m_estimator(a, y, lossfunction, clipping, preliminaryscale, regularization=tikhonov_regularization, lmbd=0):
    """
    Function that coputes the M estimator
    :param y: array that contains the measurements
    :param a: model matrix
    :param lossfunction: rho function
    :param clipping: clipping parameter for the rho function
    :param preliminaryscale: value of the scale of the residuals
    :param regularization: type of regularization used (default: tikhonov)
    :param lmbd: regularization parameter (default=0)
    :return: M estimate
    """
    m, n = a.shape
    initialx = np.ones((n, 1))  # initial solution
    xhat = irls(
        a,
        y,
        lossfunction,
        clipping,
        scale=preliminaryscale,
        initial_x=initialx,
        kind='M',
        regularization=regularization,
        lamb=lmbd
    )

    return xhat


def tau_derivative(a, y, x, clipping_1, clipping_2):
    '''
    Returns the derivative of the tau function f(x) for the given parameters y = ax + n
    Derovative defined in page 9 of
    'The regularized tau estimator: a robust and efficient solution to ill-posed problems'

    :param a: model matrix
    :param y: measurement vector
    :param x: unknown value
    :param clipping_1: clipping parameter for the rho_1 function (robust one)
    :param clipping_2: clipping paramter for the rho-2 function (non-robust one)
    :return: f'(x)

    '''
    # Get number of measurements
    m = a.shape[0]
    # get the normalized residuals
    residuals = y.reshape(-1, 1) - np.dot(a, x).reshape(-1, 1)
    scale = util.mscaleestimator(residuals, tolerance=1e-5, b=0.5, clipping=clipping_1, kind='optimal')
    rhat = np.array(residuals / scale).flatten()

    # weights for the tau rho function - independent of i
    wm = util.tauweights(rhat, lossfunction='optimal', clipping=[clipping_1, clipping_2])

    # derivative (for the moment no regularization)
    der = 0
    for element in range(m):
        # contribution to the sum of column element
        contribution = (- 1 / m) * (wm *
                                    util.scorefunction(rhat[element], kind='optimal', clipping=clipping_1
                                                       ) + util.scorefunction(rhat[element], kind='optimal',
                                                                              clipping=clipping_2)) * scale * a[element,
                                                                                                              :].T

        # add the contribution to the total sum
        der += contribution

    return der


def tau_gradient_descent(a, y, clipping_1, clipping_2, initial_x):
    '''
    Gradient descent algorithm to find the minima of the tau function
    :param a: model matrix
    :param y: measurement vector
    :param x: unknown value
    :param clipping_1: clipping parameter for the rho_1 function (robust one)
    :param clipping_2: clipping paramter for the rho-2 function (non-robust one)
    :param initial_x: initial starting point for the algorithm
    :return: local minimum
    '''

    # initialization point
    # x_init = np.random.rand(a.shape[1], 1)
    x_init = initial_x

    # number of iterations
    iter = 1000

    # step in each iteration
    step = 2

    # initialized gradient descent
    x = x_init

    for iteration in range(iter):
        # gradient descent step
        dr = tau_derivative(a, y, x, clipping_1, clipping_2)

        x -= step * tau_derivative(a, y, x, clipping_1, clipping_2).reshape(-1, 1)
        plt.scatter(x[0], x[1])

    plt.xlabel('coordinate 0')
    plt.ylabel('coordinate 1')

    return x


def soft_thresholding(v, threshold):
    """
    soft thresholding function
    :param v: scalar
    :param threshold: positive scalar
    :return: st(v)

    """

    if v >= threshold:
        return (v - threshold)
    elif v <= - threshold:
        return (v + threshold)
    else:
        return 0


def proximal_operator_l2(v, step_size):
    """
    Function to compute the proximal operator with g(x) = lambda||x||_2^2
    :param v: argument of the proximal operator
    :param step_size: parameter for the proxima operator
    :return value of the proximal operator
    """

    x = v / (2 * step_size + 1)
    # if np.linalg.norm(v) >= step_size:
    #     x = (1 - step_size/np.linalg.norm(v)) * v
    # else:
    #     x = np.zeros(v.shape)

    return x


def proximal_operator_l1(v, step_size):
    """
    Computes the proximal operator of the l1 norm, g(x) = lambda||x||_1
    :param v: argument for the proximal operator (vector)
    :param step_size: parameter in the prox. operator (lambda in the paper)
    :param lmbd: regularization parameter
    :return: value of the proximal operator
    """

    return np.sign(v) * np.maximum(np.abs(v) - step_size, 0.)


def line_search(a, y, x_k, init_step, lmbd, clipping_1, clipping_2, proximal_operator):
    """
    Given x_k, we find the step size for each iteration
    :param a: model matrix
    :param y: measurement vector
    :param x_k: x at iteration k
    :param init_step: initialization of the step
    :param lmbd: regularization parameter
    :param clipping_1: clipping parameter for the robust rho function
    :param clipping_2: clipping parameter for the efficient rho function
    :param proximal_operator: function to compute proximal operator (options: proximal_operator_l2 and
    proximal_operator_l2)
    :return: suitable step size
    """
    # Find suitable step size
    step_size = init_step  # initial guess

    # f'(x_k)
    grad_fk = tau_derivative(a, y, x_k, clipping_1, clipping_2)

    # r(x_k)
    r_k = y.reshape(-1, 1) - np.dot(a, x_k).reshape(-1, 1)

    # f(x_k)
    fk = util.tauscale(r_k, 'optimal', clipping=[clipping_1, clipping_2], b=0.5)
    while True:  # adjust step size
        xk_grad = x_k - step_size * grad_fk

        # proximal operator with x_k
        z = proximal_operator(xk_grad, step_size * lmbd)

        # residual for z
        r_values = y.reshape(-1, 1) - np.dot(a, z).reshape(-1, 1)

        # left part of the inequality, f(z)
        lhand = util.tauscale(r_values, 'optimal', clipping=[clipping_1, clipping_2], b=0.5)

        # right part of the inequality, majorization function
        rhand = fk + grad_fk.dot(z - x_k) + (0.5 * (1 / step_size)) * (z - x_k).dot(z - x_k)

        if lhand <= rhand:
            # if f(z) is smaller than the majorization function, the step size is good
            break
        else:
            # we update the step size, and check again
            step_size *= .5

    return step_size


def tau_apg(a, y, lmbd, clipping_1, clipping_2, initial_x, proximal_operator, rtol=0.0000000001):
    """
    Gradient descent algorithm to find the minima of the tau function
    :param a: model matrix
    :param y: measurement vector
    :param lmbd: regularization parameter
    :param clipping_1: clipping parameter for the rho_1 function (robust one)
    :param clipping_2: clipping paramter for the rho-2 function (non-robust one)
    :param initial_x: initial starting point for the algorithm
    :param proximal_operator: function for the proximal operator (proximal_operator_l2 or proximal_operator_l1)
    :param rtol: tol to stop the algorithm
    :return: local minimum
    """

    # initialization point
    # x_init = np.random.rand(a.shape[1], 1)
    x_init = initial_x

    # number of iterations
    iter = 100

    # initialized z and x to the same value
    x = [x_init, x_init]  # x_(k-1) and x_k
    z = x_init

    # initialize parameters for the acceleration extension
    t = [0, 1]  # t_(k-1) and t_k

    # initialize the step for prox. gradient
    default_step_size = 100

    for iteration in range(1, iter):

        # algo step without acceleration (Eq. 12)


        # parameter for the proximal operator
        alpha_x = line_search(a, y, x[1], default_step_size, lmbd, clipping_1, clipping_2, proximal_operator)

        # argument for the proximal operator
        arg_v = x[1] - alpha_x * tau_derivative(a, y, x[1], clipping_1, clipping_2)

        # update v
        v = proximal_operator(arg_v, alpha_x * lmbd)

        # extension for the acceleration (Eq. 10)

        l = x[1] + (t[0] / t[1]) * (np.array(z) - x[1]) + ((t[0] - 1) / t[1]) * (x[1] - np.array(x[0]))

        # algo step with acceleration (Eq. 11)

        # parameter for the proximal operator
        alpha_l = line_search(a, y, l, default_step_size, lmbd, clipping_1, clipping_2, proximal_operator)

        # argument for the proximal operator
        arg_l = l - alpha_l * tau_derivative(a, y, l, clipping_1, clipping_2)

        # update z
        z = proximal_operator(arg_l, alpha_l * lmbd)


        # update t
        t[0] = t[1]
        t[1] = 0.5 * (np.sqrt(4 * t[0] ** 2 + 1) + 1)

        # update x (Eq. 14)
        x[0] = x[1]

        # compare objective functions
        res_z = y.reshape(-1, 1) - np.dot(a, z).reshape(-1, 1)
        tscale_z = util.tauscale(res_z, 'optimal', [clipping_1, clipping_2], b=0.5)
        F_z = tscale_z + lmbd * np.dot(z, z)

        res_v = y.reshape(-1, 1) - np.dot(a, v).reshape(-1, 1)
        tscale_v = util.tauscale(res_v, 'optimal', [clipping_1, clipping_2], b=0.5)
        F_v = tscale_v + lmbd * np.dot(v, v)

        # take the best one
        if F_z <= F_v:
            x[1] = z
        else:
            x[1] = v

        # sct = x[1]
        # plt.scatter(sct[0], sct[1])

        # Check if the algorithm already converged
        res_old = y.reshape(-1, 1) - np.dot(a, x[0]).reshape(-1, 1)
        res_new = y.reshape(-1, 1) - np.dot(a, x[1]).reshape(-1, 1)

        F_old = util.tauscale(res_old, 'optimal', [clipping_1, clipping_2], b=0.5) + lmbd * np.dot(x[0], x[0])
        F_new = util.tauscale(res_new, 'optimal', [clipping_1, clipping_2], b=0.5) + lmbd * np.dot(x[1], x[1])
        # print 'x_k = ', x[1]

        # print 'Shrink obj func ', np.abs(F_old - F_new)
        # print 'x ', x[1]

        if (np.abs(F_old - F_new) / F_new) < rtol:
            print("Achieved relative tolerance at iteration %s" % iteration)
            break
        xdis = np.linalg.norm(x[0] - x[1])
        if xdis < rtol:
            print("Achieved relative tolerance at iteration %s" % iteration)
            break

    # plt.show()
    obj_distance = np.abs(F_old - F_new)
    # print 's_z = ', tscale_z
    # print 's_v = ', tscale_v
    # print 'z = ', z
    # print 'v = ', v

    return x[1], obj_distance, alpha_l, alpha_x


def basictau(
        a,
        y,
        loss_function,
        clipping,
        ninitialx,
        maxiter=100,
        nbest=1,
        initialx=None,
        b=0.5,
        regularization=tikhonov_regularization,
        lamb=0
):
    '''
    This routine minimizes the objective function associated with the tau-estimator.
    For more information on the tau estimator see http://arxiv.org/abs/1606.00812

    This function is hard to minimize because it is non-convex. This means that it has several local minima. Depending on
    the initial x that we use for our minimization, we will end up in a different local minimum.

    In this algorithm we take the 'brute force' approach: let's try many different initial solutions, and let's pick the
    minimum with smallest value. The output of basictau are the best nbest minima (we will need them later)

    :param a: matrix A in y - Ax
    :param y: vector y in y - Ax
    :param loss_function: type of the rho function we are using
    :param clipping: clipping parameters. In this case we need two, because the rho function for the tau is composed two rho functions.
    :param ninitialx: how many different solutions do we want to use to find the global minimum (this function is not convex!)
                      if ninitialx=0, means the user introduced a predefined initial solution
    :param maxiter: maximum number of iteration for the irls algorithm
    :param nbest: we return the best nbest solutions. This will be necessary for the fast algorithm
    :param initialx: the user can define here the initial x he wants
    :param b: this is a parameter to estimate the scale
    :param regularization: function for the regularization we want to use. Default: tikhonov_regularization
    :param lamb: parameter for the regularization. It has to be non-negative.

    :return xhat: contains the best nmin estimations of x
    :return mintauscale: value of the objective function when x = xhat
    '''

    # TODO : remove this, for testing only
    np.set_printoptions(precision=20)


    # to store the minimum values of the objective function (in this case is
    # the scale)
    mintauscale = np.empty((nbest, 1))

    # initializing objective function with infinite. When we have a x that gives a smaller value for the obj. function,
    # we store the value of the objective function here
    mintauscale[:] = float("inf")

    # count how many initial solutions are we trying
    k = 0

    # store here the best xhat (nbest of them)
    xhat = np.zeros((a.shape[1], nbest))  # to store the best nmin minima

    # auxiliary variable to check if the user introduced a predefined initial solution.
    # = 0 if we do not have initial x. =1 if we have a given initial x
    givenx = 0

    if initialx is None:
        initialx = np.ones(a.shape[1])

    if ninitialx == 0:
        # we have a predefined initial x
        ninitialx = initialx.shape[1]

        # set givenx to 1
        givenx = 1

    while k < ninitialx:
        # if still we did not reach the number of initial solutions that we want to try,
        if givenx == 1:
            # if we have a given initial solution initx, we take it
            initx = np.expand_dims(initialx[:, k], axis=1)
        else:
            # otherwise get randomly  a new initial solution initx
            # TODO : remove this line and put back initx = util.getinitialsolution(y.reshape(-1, 1), a)
            initx = np.ones(a.shape[1])
            # initx = util.getinitialsolution(y.reshape(-1, 1), a)

        # compute the residual y - Ainitx
        initialres = y.reshape(-1, 1) - np.dot(a, initx)

        # TODO : added by Guillaume
        initialres = np.array(initialres)

        # estimate the scale using initialres
        initials = np.median(np.abs(initialres)) / 0.6745

        # print 'MARTA initial_residuals = ', initialres
        # print 'MARTA scale = ', initials
        # print 'Marta initial x = ', initx

        # solve irls using y, a, the tau weights, initx and initals. We get an
        # estimation of x, xhattmp
        xhattmp = irls(
            a,
            y,
            loss_function,
            clipping,
            scale=initials,
            initial_x=initx,
            kind='tau',
            b=0.5,
            max_iterations=maxiter,
            regularization=regularization,
            lamb=lamb)

        # print 'MARTA IRLS RESULT = ', xhattmp
        # last working version
        # xhattmp, scaletmp, ni, w, steps = inv.irls(y, a, 'tau', 'optimal', 'none', 0, initx, initials, clipping, maxiter)

        # print 'Marta y = ', y
        #
        # print 'Marta a * xhattmp = ', np.dot(a, xhattmp).reshape(-1, 1)

        # compute the value of the objective function using xhattmp
        # we compute the res first
        res = y.reshape(-1, 1) - np.dot(a, xhattmp).reshape(-1, 1)
        # res = np.subtract(y.reshape(-1, 1), np.dot(a, xhattmp).reshape(-1, 1))



        # Value of the objective function using xhattmp
        # tscalesquare = util.tauscale(res, lossfunction, clipping, b)

        # print 'Marta residuals = ', res

        tscalesquare = util.tauscale(res, 'optimal', clipping, b)

        # print 'Marta tscalesquare = ', tscalesquare

        # update counter
        k += 1

        # we checks if the objective function has a smaller value that then
        # ones we found before
        if tscalesquare < np.amax(mintauscale):
            # it is smaller, so we keep it!
            # store value for the objective function
            mintauscale[np.argmax(mintauscale)] = tscalesquare

            # store value of xhat
            xhat[:, np.argmax(mintauscale)] = np.squeeze(xhattmp)

    # we return the best solutions we found, with the value of the objective
    # function associated with the xhats
    return xhat, mintauscale


def fasttau(y,
            a,
            loss_function,
            clipping,
            ninitialx,
            regularization=tikhonov_regularization,
            nmin=5,
            initialiter=5,
            lamb=0
            ):
    '''
    Fast version of the basic tau algorithm.
    To save some computational cost, this algorithm exploits the speed of convergence of the basic algorithm.
    It has two steps: in the first one, for every initial solution, it only performs initialiter iterations. It keeps
    value of the objective function.
    In the second step, it compares the value of all the objective functions, and it select the nmin smaller ones.
    It iterates them until convergence. Finally, the algorithm select the x that produces the smallest objective
    function.
    For more details see http://arxiv.org/abs/1606.00812

    :param a: matrix A in y - Ax
    :param y: vector y in y - Ax
    :param loss_function: type of the rho function we are using
    :param clipping: clipping parameters. In this case we need two, because the rho function for the tau is composed two rho functions.
    :param ninitialx: how many different solutions do we want to use to find the global minimum (this function is not convex!)
                    if ninitialx=0, means the user introduced a predefined initial solution
    :param regularization: function for the regularization we want to use. Default: tikhonov_regularization
    :param nmin: how many solutions do we keep for the second round.
    :param initialiter: number of iterations during the first round
    :param lamb: parameter for the regularization. It has to be non-negative.

    :return xhat: contains the best nmin estimations of x
    :return mintauscale: value of the objective function when x = xhat
    '''
    xhat, mintauscale = basictau(
        a=a,
        y=y,
        loss_function=loss_function,
        clipping=clipping,
        ninitialx=ninitialx,
        maxiter=initialiter,
        nbest=nmin,
        regularization=regularization,
        lamb=lamb
    )  # first round: only initialiter iterations. We keep the nmin best solutions

    xfinal, tauscalefinal = basictau(
        a=a,
        y=y,
        loss_function=loss_function,
        clipping=clipping,
        regularization=regularization,
        ninitialx=0,
        maxiter=100,
        nbest=1,
        initialx=xhat,
        lamb=lamb
    )  # iterate the best solutions until convergence

    return xfinal, tauscalefinal


# function to apply a loss function to any data structure; scalar, array or matrix
# returns the exact same data structure with the loss function applied element-wise
def array_loss(values, loss_function, clipping=None):
    # if we input incorrect values (due to division by too tiny numbers in irls)
    # inside array_loss, it returns zeroes

    # checks if there is a 'nan' value in the input
    if np.any(np.isnan(values)):
        print "incorrect input in array loss !"
        return np.zeros(values.shape)

    kwargs = {}
    if clipping != None:
        kwargs['clipping'] = clipping

    vfunc = np.vectorize(loss_function)
    return vfunc(values, **kwargs)


def tau_weights_new(input, clipping, loss_function=rho_optimal):
    '''
    This is the beginning of the refactoring of the whole tau estimator. We did not have to finish it, so we used the
    architecture...
    :return array or float: element-wise result of f(x)/x if x!=0, 0 otherwise
    :param input: (array or float) vector or float to be processed, x_i's
    :param clipping: clipping parameter for loss function -- they should be two!
    :param loss_function (optional): family of loss functions we use
    :return: ndarray with tau weights
    '''

    weights = 2.0 * array_loss(input, loss_function, clipping[1])
    weights -= (util.scoreoptimal(input, clipping[1]) * input)
    scaling = np.sum(util.scoreoptimal(input, clipping[0]) * input)

    if (scaling == 0):
        scaling = 1.0

    return weights / scaling


def sridge_matlab(A, y):
    '''
    This function runs the sridge function written in Matlab code.
    :param A: model matrix
    :param y: measurement vector
    :return: S-estimat
    '''
    # import Sridge.matlab as sr
    # build the path for the matlab code
    # matlab_folder = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'Sridge/matlab')
    # python_folder = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'Sridge/python')
    # # append it to the current path
    # sys.path.append(python_folder)

    # start matlab engine
    eng = matlab.engine.start_matlab()

    # call matlab function
    tf = eng.sridge(A, y, 3, 3)


# ===============================================================================
# ===================================TESTING AREA================================
# ===============================================================================

def main():
    import array
    # defining linear problem, create data
    m = 20  # number of measurements
    n = 3  # dimensions of x
    a = np.random.rand(m, n)  # model matrix
    x_grountruth = [3, 5, 1]  # ground truth
    y = np.dot(a, x_grountruth).reshape(-1, 1) + 0.5 * np.random.rand(m, 1)

    # define any x
    x = [7, 9, 3]

    # clipping parameters
    clipping_1 = 1.21
    clipping_2 = 3.27
    reg_parameter = 0.001

    # testing apg algo

    # x_apg = tau_apg(a, y, reg_parameter, clipping_1, clipping_2, x, proximal_operator_l1, rtol=1e-5)

    # print 'x_apg =', x_apg
    initx = np.array(x)
    initx = initx.reshape(-1, 1)

    # x_irls = basictau(
    #     a,
    #     y,
    #     'optimal',
    #     [clipping_1, clipping_2],
    #     ninitialx=0,
    #     maxiter=100,
    #     nbest=1,
    #     initialx=initx,
    #     b=0.5,
    #     regularization=lasso_regularization,
    #     lamb=reg_parameter
    #     )

    # print 'x_irls = ', x_irls[0]

    x_m = m_estimator(a, y, 'optimal', clipping_1, 1, regularization=lasso_regularization, lmbd=reg_parameter)

    print 'estimate x_m =', x_m


    # cast to matlab types
    # amat = matlab.double(a.tolist())

    # ymat = matlab.double(y.tolist())

    # sridge_matlab(amat, ymat)


if __name__ == '__main__':
    # To launch manually in console:
    # python plotFunctions.py
    main()
