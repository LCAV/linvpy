from __future__ import division
import numpy as np

__author__ = 'GuillaumeBeaud'


# Abstract class for loss functions so they share the same interface and all
#  have the rho, psi, weights functions
class LossFunction:
    def __init__(self, clipping=None):  # Constructor of the class
        if clipping is not None:
            assert clipping > 0  # verifies clipping is >0 if it is not None
            self.clipping = clipping

    # vectorized rho function : applies element-wise rho function to any
    # structure and returns same structure
    def rho(self, array):  # Abstract method, defined by convention only
        raise NotImplementedError("Subclass must implement abstract method")

    # vectorized rho function : applies element-wise psi function to any
    # structure and returns same structure
    def psi(self, array):  # Abstract method, defined by convention only
        raise NotImplementedError("Subclass must implement abstract method")

    # classic weights for m-estimator
    def m_weights(self, matrix):
        # operation on a single element, which is then vectorized
        # weight(e) = psi(e) / 2e or 1 if e==0
        def unit_operation(element):
            if element == 0:
                return 1.0
            else:
                return self.psi(element) / (2 * element)

        # unit_operation is vectorized to operate on a matrix
        vfunc = np.vectorize(unit_operation)

        # returns the matrix with the unit_operation executed on each element
        return vfunc(matrix)

    # Plots the rho, psi and weights on the given interval with a step of 0.1
    def plot(self, interval):
        """
        :param interval: The interval the functions will be plotted on.
        :type interval: integer
        """
        import matplotlib.pyplot as plt

        # Creates a range between -interval and interval with a step of 0.1
        x = np.arange(-interval, interval, 0.1)

        # The different functions to be plotted on the interval
        y_rho = self.rho(x)
        y_psi = self.psi(x)
        y_weights = self.m_weights(x)

        plt.plot(x, y_rho,
                 label=self.__class__.__name__ + ' rho(x)'
                 )

        plt.plot(x, y_psi,
                 label=self.__class__.__name__ + ' psi(x)'
                 )

        plt.plot(x, y_weights,
                 label=self.__class__.__name__ + ' weights(x)'
                 )

        plt.legend(
            bbox_to_anchor=(0., 1.02, 1., .102),
            loc=3, ncol=2, mode="expand", borderaxespad=0.
        )

        plt.xlabel('x')

        plt.xlim(-interval, interval)

        plt.ylim()

        plt.show()


class Huber(LossFunction):
    """
    :param clipping: Value of the clipping to be used in the loss function
    :type clipping: float
    """

    def __init__(self, clipping=1.345):
        LossFunction.__init__(self, clipping)
        if clipping is None:
            self.clipping = 1.345

    def rho(self, array):
        """
        The regular huber loss function; the "rho" version.

        :math:`\\rho(x)=\\begin{cases}
        \\frac{1}{2}{x^2}& \\text{if |x|} \\leq clipping \\\\
        clipping (|x| - \\dfrac{1}{2} clipping)& \\text{otherwise}
        \\end{cases}`

        This function is quadratic for small inputs, and linear for large
        inputs.

        :param array: Array of values to apply the loss function to
        :type array: numpy.ndarray
        :return: Array of same shape as the input, cell-wise results of the\
        loss function
        :rtype: numpy.ndarray

        :Example:

        >>> import numpy as np
        >>> import linvpy as lp

        >>> huber = lp.Huber()
        >>> huber.rho(2)
        array(1.7854875000000001)

        >>> y = np.array([1, 2, 3])
        >>> huber.rho(y)
        array([ 0.5      ,  1.7854875,  3.1304875])

        >>> a = np.matrix([[1, 2], [3, 4], [5, 6]])
        >>> huber.rho(a)
        matrix([[ 0.5      ,  1.7854875],
            [ 3.1304875,  4.4754875],
            [ 5.8204875,  7.1654875]])

        >>> # Plots the rho, psi and m_weights on the given interval
        >>> huber.plot(15)

        .. figure:: images/huber.png
        """
        # rho version of the Huber loss function
        def unit_rho(element):
            # Casts to float to avoid comparison issues
            element = float(element)
            if abs(element) <= self.clipping:
                return element ** 2 / 2.0
            else:
                return self.clipping * (abs(element) - self.clipping / 2.0)

        vfunc = np.vectorize(unit_rho)
        return vfunc(array)

    def psi(self, array):
        """
        Derivative of the Huber loss function; the "psi" version. Used in
        the weight function of the M-estimator.

        :math:`\\psi(x)=\\begin{cases}
        x& \\text{if |x|} \\leq clipping \\\\
        clipping \\cdot sign(x) & \\text{otherwise}
        \\end{cases}`

        :param array: Array of values to apply the loss function to
        :type array: numpy.ndarray
        :return: Array of same shape as the input, cell-wise results of the\
         loss function
        :rtype: numpy.ndarray

        :Example:

        >>> import numpy as np
        >>> import linvpy as lp

        >>> huber = lp.Huber()
        >>> huber.psi(2)
        array(1.345)

        >>> y = np.array([1, 2, 3])
        >>> huber.psi(y)
        array([ 1.   ,  1.345,  1.345])

        >>> a = np.matrix([[1, 2], [3, 4], [5, 6]])
        >>> huber.psi(a)
        matrix([[ 1.   ,  1.345],
            [ 1.345,  1.345],
            [ 1.345,  1.345]])
        """
        # psi version of the Huber loss function
        def unit_psi(element):
            # Casts to float to avoid comparison issues
            element = float(element)
            if abs(element) >= self.clipping:
                return self.clipping * np.sign(element)
            else:
                return element

        vfunc = np.vectorize(unit_psi)
        return vfunc(array)


class Bisquare(LossFunction):
    """
    :param clipping: Value of the clipping to be used in the loss function
    :type clipping: float
    """

    def __init__(self, clipping=4.685):
        LossFunction.__init__(self, clipping)
        if clipping is None:
            self.clipping = 4.685

    def rho(self, array):
        """
        The regular bisquare loss (or Tukey's loss), "rho" version.

        :math:`\\rho(x)=\\begin{cases}
        (\\frac{c^2}{6})(1-(1-(\\frac{x}{c})^2)^3)& \\text{if |x|} \\leq 0
        \\\\
        \\frac{c^2}{6}& \\text{if |x| > 0}
        \\end{cases}`

        :param array: Array of values to apply the loss function to
        :type array: numpy.ndarray
        :return: Array of same shape as the input, cell-wise results of the\
         loss function
        :rtype: numpy.ndarray

        :Example:

        >>> import numpy as np
        >>> import linvpy as lp

        >>> bisquare = lp.Bisquare()
        >>> bisquare.rho(2)
        array(1.6576630874988754)

        >>> y = np.array([1, 2, 3])
        >>> bisquare.rho(y)
        array([ 0.4775661 ,  1.65766309,  2.90702817])

        >>> a = np.matrix([[1, 2], [3, 4], [5, 6]])
        >>> bisquare.rho(a)
        matrix([[ 0.4775661 ,  1.65766309],
            [ 2.90702817,  3.58536054],
            [ 3.65820417,  3.65820417]])

        >>> # Plots the rho, psi and m_weights on the given interval
        >>> bisquare.plot(15)

        .. figure:: images/bisquare.png

        """
        # rho version of the Bisquare loss function
        def unit_rho(element):
            # Casts to float to avoid comparison issues
            element = float(element)
            if abs(element) <= self.clipping:
                return ((self.clipping ** 2.0) / 6.0) * \
                       (1 - (1 - (element / self.clipping) ** 2) ** 3)
            else:
                return (self.clipping ** 2) / 6.0

        vfunc = np.vectorize(unit_rho)
        return vfunc(array)

    def psi(self, array):
        """
        The derivative of bisquare loss (or Tukey's loss), "psi" version.

        :math:`\\psi(x)=\\begin{cases}
        x((1-(\\frac{x}{c})^2)^2)& \\text{if |x|} \\leq 0 \\\\
        0& \\text{if |x| > 0}
        \\end{cases}`

        :param array: Array of values to apply the loss function to
        :type array: numpy.ndarray
        :return: Array of same shape as the input, cell-wise results of the\
         loss function
        :rtype: numpy.ndarray

        :Example:

        >>> import numpy as np
        >>> import linvpy as lp

        >>> bisquare = lp.Bisquare()
        >>> bisquare.psi(2)
        array(1.3374668237772656)

        >>> y = np.array([1, 2, 3])
        >>> bisquare.psi(y)
        array([ 0.9109563 ,  1.33746682,  1.04416812])

        >>> a = np.matrix([[1, 2], [3, 4], [5, 6]])
        >>> bisquare.psi(a)
        matrix([[ 0.9109563 ,  1.33746682],
            [ 1.04416812,  0.2938613 ],
            [ 0.        ,  0.        ]])
        """
        # psi version of the Bisquare loss function
        def unit_psi(element):
            # Casts to float to avoid comparison issues
            element = float(element)
            if abs(element) <= self.clipping:
                return element * ((1 - (element / self.clipping) ** 2) ** 2)
            else:
                return 0.0

        vfunc = np.vectorize(unit_psi)
        return vfunc(array)


class Cauchy(LossFunction):
    """
    :param clipping: Value of the clipping to be used in the loss function
    :type clipping: float
    """

    def __init__(self, clipping=2.3849):
        LossFunction.__init__(self, clipping)
        if clipping is None:
            self.clipping = 2.3849

    def rho(self, array):
        """
        :param array: Array of values to apply the loss function to
        :type array: numpy.ndarray
        :return: Array of same shape as the input, cell-wise results of the\
         loss function
        :rtype: numpy.ndarray

        :Example:

        >>> import numpy as np
        >>> import linvpy as lp

        >>> cauchy = lp.Cauchy()
        >>> cauchy.rho(2)
        array(1.5144982928548816)

        >>> y = np.array([1, 2, 3])
        >>> cauchy.rho(y)
        array([ 0.46060182,  1.51449829,  2.69798124])

        >>> a = np.matrix([[1, 2], [3, 4], [5, 6]])
        >>> cauchy.rho(a)
        matrix([[ 0.46060182,  1.51449829],
            [ 2.69798124,  3.80633511],
            [ 4.79348926,  5.66469239]])

        >>> # Plots the rho, psi and m_weights on the given interval
        >>> cauchy.plot(15)

        .. figure:: images/cauchy.png
        """
        # rho version of the Cauchy loss function
        def unit_rho(element):
            # Casts to float to avoid comparison issues
            element = float(element)
            return (self.clipping ** 2 / 2) * np.log(
                1 + (element / self.clipping) ** 2)

        vfunc = np.vectorize(unit_rho)
        return vfunc(array)

    def psi(self, array):
        """
        :param array: Array of values to apply the loss function to
        :type array: numpy.ndarray
        :return: Array of same shape as the input, cell-wise results of the\
         loss function
        :rtype: numpy.ndarray

        :Example:

        >>> import numpy as np
        >>> import linvpy as lp

        >>> cauchy = lp.Cauchy()
        >>> cauchy.psi(2)
        array(1.1742146893434733)

        >>> y = np.array([1, 2, 3])
        >>> cauchy.psi(y)
        array([ 0.85047284,  1.17421469,  1.16173317])

        >>> a = np.matrix([[1, 2], [3, 4], [5, 6]])
        >>> cauchy.psi(a)
        matrix([[ 0.85047284,  1.17421469],
            [ 1.16173317,  1.0490251 ],
            [ 0.92671316,  0.81862153]])
        """
        # psi version of the Cauchy loss function
        def unit_psi(element):
            # Casts to float to avoid comparison issues
            element = float(element)
            return element / (1 + (element / self.clipping) ** 2)

        vfunc = np.vectorize(unit_psi)
        return vfunc(array)


class Optimal(LossFunction):
    """
    :param clipping: Value of the clipping to be used in the loss function
    :type clipping: float
    """

    def __init__(self, clipping=3.270):
        LossFunction.__init__(self, clipping)
        if clipping is None:
            self.clipping = 3.270

    def rho(self, array):
        """
        :param array: Array of values to apply the loss function to
        :type array: numpy.ndarray
        :return: Array of same shape as the input, cell-wise results of the\
         loss function
        :rtype: numpy.ndarray

        :Example:

        >>> import numpy as np
        >>> import linvpy as lp

        >>> optimal = lp.Optimal()
        >>> optimal.rho(2)
        array(0.057550768770363074)

        >>> y = np.array([1, 2, 3])
        >>> optimal.rho(y)
        array([ 0.01438769,  0.05755077,  0.12948923])

        >>> a = np.matrix([[1, 2], [3, 4], [5, 6]])
        >>> optimal.rho(a)
        matrix([[ 0.01438769,  0.05755077],
            [ 0.12948923,  0.23020308],
            [ 0.3596923 ,  0.51795692]])

        >>> # Plots the rho, psi and m_weights on the given interval
        >>> optimal.plot(15)

        .. figure:: images/optimal.png
        """
        # rho version of the Optimal loss function
        def unit_rho(element):
            # Casts to float to avoid comparison issues
            element = float(element)
            y = abs(element / self.clipping)
            if y <= 2.0:
                return y ** 2 / 2.0 / 3.25
            elif 2 < y <= 3:
                return (1.792 - 0.972 * y ** 2 + 0.432 * y ** 4 -
                        0.052 * y ** 6 + 0.002 * y ** 8) / 3.25
            else:
                return 1.0

        vfunc = np.vectorize(unit_rho)
        return vfunc(array)

    def psi(self, array):
        """
        :param array: Array of values to apply the loss function to
        :type array: numpy.ndarray
        :return: Array of same shape as the input, cell-wise results of the\
         loss function
        :rtype: numpy.ndarray

        :Example:

        >>> import numpy as np
        >>> import linvpy as lp

        >>> optimal = lp.Optimal()
        >>> optimal.psi(2)
        array(0.05755076877036309)

        >>> y = np.array([1, 2, 3])
        >>> optimal.psi(y)
        array([ 0.02877538,  0.05755077,  0.08632615])

        >>> a = np.matrix([[1, 2], [3, 4], [5, 6]])
        >>> optimal.psi(a)
        matrix([[ 0.02877538,  0.05755077],
            [ 0.08632615,  0.11510154],
            [ 0.14387692,  0.17265231]])

        """
        # psi version of the Optimal loss function
        def unit_psi(element):
            # Casts to float to avoid comparison issues
            element = float(element)
            y = abs(element)
            if y <= 2.0 * self.clipping:
                return element / self.clipping ** 2 / 3.25
            elif 2.0 * self.clipping < y <= 3 * self.clipping:
                return (
                           -1.944 * element / self.clipping ** 2 + 1.728 *
                           element ** 3 /
                           self.clipping ** 4 - 0.312 * element ** 5 /
                           self.clipping ** 6 +
                           0.016 * element ** 7 / self.clipping ** 8) / 3.25
            else:
                return 0.0

        vfunc = np.vectorize(unit_psi)
        return vfunc(array)


# Abstract class for regularization functions so they share the same interface
class Regularization:
    # no constructor needed. This class is used as an interface for all the
    # regularization functions
    def __init__(self):
        pass

    # Abstract method, defined by convention only.
    # Subclasses of Regularization must implement this function.
    def regularize(self, a, y):
        raise NotImplementedError("Subclass must implement abstract method")


class Tikhonov(Regularization):
    """
    The standard approach to solve the problem :math:`\\mathbf{y = Ax +
    n}` explained above is to use the  ordinary least squares method.
    However if your matrix :math:`\\mathbf{A}` is a fat matrix (it has more
    columns than rows) or it has a large condition number, then you should
    use a regularization to your problem in order to get a meaningful
    estimation of :math:`\\mathbf{x}`.

    The Tikhonov regularization is a tradeoff between the least squares
    solution and the minimization of the L2-norm of the output :math:`x` (
    L2-norm = sum of squared values of the vector :math:`x`), :math:`\\hat{
    \\mathbf{x}} = {\\rm arg}\\min_x\\,\\lVert \\mathbf{y - Ax} \\rVert_2^2
    + \\lambda\\lVert \\mathbf{x} \\rVert_2^2`

    The parameter lambda tells how close to the least squares solution the
    output :math:`\\mathbf{x}` will be; a large lambda will make
    :math:`\\mathbf{x}` close to :math:`\\lVert\\mathbf{x}\\rVert_2^2 = 0`,
    while a small lambda will approach the least squares solution (Running
    the function with lambda=0 will behave like the ordinary least_squares()
    method).

    The Tikhonov solution has an analytic solution and it is given by
    :math:`\\hat{\\mathbf{x}} = (\\mathbf{A^{T}A}+ \\lambda^{2} \\mathbf{
    I})^{-1}\\mathbf{A}^{T}\\mathbf{y}`, where :math:`\\mathbf{I}` is the
    identity matrix.
    """
    pass

    # returns the Tikhonov regularization from A,y,lambda
    def regularize(self, a, y, lamb=0):
        """
        :param a: MxN matrix A in the y=Ax equation
        :type a: numpy.ndarray
        :param y: M vector y in the y=Ax equation
        :type y: numpy.ndarray
        :param lamb: non-negative tradeoff parameter between least squares\
         and minimization of the L-2 norm
        :type lamb: integer
        :return: N vector x in the y=Ax equation
        :rtype: numpy.ndarray

        :Example:

        >>> import numpy as np
        >>> import linvpy as lp

        >>> a = np.matrix([[1, 2], [3, 4], [5, 6]])
        >>> y = np.array([1, 2, 3])

        >>> tiko = lp.Tikhonov()

        >>> tiko.regularize(a, y, 2)
        array([ 0.21782178,  0.30693069])

        >>> tiko.regularize(a, y)
        array([ -5.97106181e-17,   5.00000000e-01])
        """
        assert lamb >= 0
        y = np.squeeze(np.asarray(y))  # flattens y into a vector

        # if lambda == 0 it simply returns the least squares solution
        if lamb == 0:
            return np.linalg.lstsq(a, y)[0].reshape(-1)
        else:
            identity_matrix = np.identity(a.shape[1])
            # output = (A' A + lambda^2 I)^-1 A' y
            xhat = np.dot(
                np.dot(
                    np.linalg.inv(
                        np.add(
                            np.dot(a.T, a),
                            np.dot(lamb ** 2, identity_matrix)
                        ),
                    ), a.T
                ), y)
            return np.squeeze(
                np.asarray(xhat))  # flattens result into an array


class Lasso(Regularization):
    """
    Lasso algorithm that solves :math:`min ||\\mathbf{y - Ax}||_2^2 + lambda
    ||x||_1`
    """
    pass

    # returns the Lasso regularization from A,y,lambda
    def regularize(self, a, y, lamb=0):
        """
        :param a: MxN matrix A in the y=Ax equation
        :type a: numpy.ndarray
        :param y: M vector y in the y=Ax equation
        :type y: numpy.ndarray
        :param lamb: non-negative regularization parameter
        :type lamb: integer
        :return: N vector x in the y=Ax equation
        :rtype: numpy.ndarray

        :Example:

        >>> import numpy as np
        >>> import linvpy as lp

        >>> a = np.matrix([[1, 2], [3, 4], [5, 6]])
        >>> y = np.array([1, 2, 3])

        >>> lasso = lp.Lasso()

        >>> lasso.regularize(a, y, 2)
        array([ 0.        ,  0.48214286])

        >>> lasso.regularize(a, y)
        array([ -5.97106181e-17,   5.00000000e-01])
        """
        from sklearn import linear_model

        assert lamb >= 0

        # if lambda == 0 it simply returns the least squares solution
        if lamb == 0:
            return np.squeeze(np.asarray(np.linalg.lstsq(a, y)[0].reshape(-1)))

        # Converts regularization parameter (sklearn considers (1/2m factor))
        reg_parameter = lamb / (2 * len(y))

        # Initialize model
        clf = linear_model.Lasso(reg_parameter, fit_intercept=False,
                                 normalize=False)

        # Fit it
        clf.fit(a, y)

        # Returns estimate
        x = clf.coef_

        return x


# Super class of the M and Tau Estimators. All values are default so you can
#  simply create one
# with my_estimator = MEstimator() and then my_estimator.estimate(A,y) which
#  gives the answer.
class Estimator:
    def __init__(self,
                 loss_function=Huber,
                 clipping=None,
                 regularization=Tikhonov(),
                 lamb=0,
                 scale=1.0,
                 b=0.5,
                 tolerance=1e-5,
                 max_iterations=100):
        assert scale != 0
        self.loss_function = loss_function(clipping=clipping)
        self.regularization = regularization
        self.lamb = lamb
        self.scale = scale
        self.b = b
        self.tolerance = tolerance
        self.max_iterations = max_iterations

    # Iteratively re-weighted least squares
    def irls(self, a, y, initial_x):

        # if an initial value for x is specified, use it, otherwise generate
        #  a vector of ones
        if initial_x is not None:
            vector_x = initial_x
        else:
            # Generates a ones vector_x with length = A.columns
            vector_x = np.ones(a.shape[1])
            initial_x = np.ones(a.shape[1])

        # Ensures numpy types and flattens y into array
        a = np.matrix(a)
        y = np.matrix(y)
        y = y.reshape(-1, 1)
        initial_x = initial_x.reshape(-1, 1)

        # Residuals = y - Ax, difference between measured values and model
        residuals = y - np.dot(a, initial_x).reshape(-1, 1)

        for i in range(1, self.max_iterations):

            # This "if" tests whether the object calling irls is a Tau- or a
            #  M-Estimator
            # In case of Tau-Estimator, we need to update the estimation of
            # the scale in each iteration
            if isinstance(self, TauEstimator):

                # Flattens the residuals
                residuals = np.asarray(residuals.reshape(-1)).flatten()

                # scale = scale * (mean(loss_function(residuals/scale))/b)^1/2
                self.scale *= np.sqrt(
                    np.mean(
                        self.loss_function_1.rho(residuals / self.scale)
                    ) / self.b
                )

                # if the scale is 0 we have a good enough solution so we
                # return the current x
                if self.scale == 0.0:
                    return vector_x

                # normalize residuals : rhat = ((y - Ax)/ self.scale)
                rhat = np.array(residuals / self.scale).flatten()

                # # computes the weights for tau
                z = self.score_function(rhat)

                # first weights are set to an array of ones
                weights_vector = np.ones(rhat.shape)

                # returns the positions of the nonzero elements of rhat
                i = np.nonzero(rhat)

                # weights = score_function(rhat) / (2 * A.shape[0] * rhat)
                # for nonzero elements of rhat
                # weights = 1 otherwise
                weights_vector[i] = z[i] / (2 * a.shape[0] * rhat[i])

            # If the object calling irls is not a Tau-Estimator we use the
            # normal weights
            else:
                # normalize residuals : rhat = ((y - Ax)/ self.scale)
                rhat = np.array(residuals / self.scale).flatten()

                # weights_vector = weights of rhat according to the loss
                # function
                weights_vector = self.loss_function.m_weights(rhat)

            # Makes a diagonal matrix with the values of the weights_vector
            # np.squeeze(np.asarray()) flattens the matrix into a vector
            weights_matrix = np.diag(
                np.squeeze(
                    np.asarray(weights_vector)
                )
            )

            # Square root of the weights matrix, sqwm = W^1/2
            sqwm = np.sqrt(weights_matrix)

            # a_weighted = W^1/2 A
            a_weighted = np.dot(sqwm, a)

            # y_weighted = diagonal of W^1/2 y
            y_weighted = np.dot(sqwm, y)

            # vector_x_new is there to keep the previous value to compare
            vector_x_new = self.regularization.regularize(a_weighted,
                                                          y_weighted,
                                                          self.lamb)

            # Distance between previous and current iteration
            xdis = np.linalg.norm(vector_x - vector_x_new)

            # New residuals
            residuals = y.reshape(-1) - np.dot(a, vector_x_new).reshape(-1)

            # Divided by the specified optional self.scale, otherwise
            # self.scale = 1
            vector_x = vector_x_new

            # if the difference between iteration n and iteration n+1 is
            # smaller than self.tolerance, return vector_x
            if xdis < self.tolerance:
                return vector_x

        return vector_x

    # Abstract method so all estimators have a Estimator.estimate function
    def estimate(self, a, y):
        raise NotImplementedError("Subclass must implement abstract method")


# Inherits every feature from the class Estimator
class MEstimator(Estimator):
    """
    The M-estimator uses the method of iteratively reweighted least squares \
     (IRLS) to minimize iteratively the function:

    :math:`\\boldsymbol x^{(t+1)} =`
    :math:`{\\rm arg}\\min_x\\,`
    :math:`\\big|\\big| \\boldsymbol W (\\boldsymbol x^{(t)})(\\boldsymbol\
     y - \\boldsymbol A \\boldsymbol x )\\big | \\big |_2^2.`

    The IRLS is used, among other things, to compute the M-estimate and the \
    tau-estimate.

    :param loss_function:  loss function to be used in the estimation
    :type loss_function: linvpy.LossFunction type
    :param clipping: clipping to be used in the loss function
    :type clipping: float
    :param regularization: regularization function to regularize the y=Ax \
    system
    :type regularization: linvpy.Regularization type
    :param lamb: lambda to be used in the regularization (lambda = 0 is \
    equivalent to using least squares)
    :type lamb: integer
    :param scale:
    :type scale: float
    :param b:
    :type b: float
    :param tolerance: treshold : when residuals < tolerance, the current \
     solution is returned
    :type tolerance: float
    :param max_iterations: maximum number of iterations of the iteratively \
     reweighted least squares
    :type max_iterations: integer
    """
    pass

    # The estimate function for the M-Estimator simply returns the irls
    # solution
    def estimate(self, a, y, initial_x=None):
        """
        :param a: MxN matrix A in the y=Ax equation
        :type a: numpy.ndarray
        :param y: M vector y in the y=Ax equation
        :type y: numpy.ndarray
        :param initial_x: N vector of an initial solution
        :type initial_x: numpy.ndarray
        :return: best estimation of the N vector x in the y=Ax equation
        :rtype: numpy.ndarray

        :Example:

        >>> import numpy as np
        >>> import linvpy as lp

        >>> a = np.matrix([[1, 2], [3, 4], [5, 6]])
        >>> y = np.array([1, 2, 3])

        >>> m = lp.MEstimator()
        >>> m.estimate(a,y)
        array([ -2.95552481e-16,   5.00000000e-01])

        >>> m_ = lp.MEstimator(loss_function=lp.Bisquare, clipping=2.23, \
        regularization=lp.Lasso(), lamb=3)
        >>> initial_solution = np.array([1, 2])
        >>> m_.estimate(a, y, initial_x=initial_solution)
        array([ 0.,  0.])
        """
        return self.irls(a, y, initial_x)

class TauEstimator(Estimator):
    """
    Description of the tau-estimator

    :param loss_function: loss function to be used in the estimation
    :type loss_function: linvpy.LossFunction type
    :param clipping_1: first clipping value of the loss function
    :type clipping_1: float
    :param clipping_2: second clipping value of the loss function
    :type clipping_2: float
    :param regularization: regularization function to regularize the y=Ax system
    :type regularization: linvpy.Regularization type
    :param lamb: lambda to be used in the regularization (lambda = 0 is\
     equivalent to using least squares)
    :type lamb: integer
    :param scale:
    :type scale: float
    :param b:
    :type b: float
    :param tolerance: treshold : when residuals < tolerance, the current\
     solution is returned
    :type tolerance: float
    :param max_iterations: maximum number of iterations of the iteratively\
     reweighted least squares
    :type max_iterations: integer
    """

    def __init__(self,
                 loss_function=Huber,
                 clipping_1=None,
                 clipping_2=None,
                 regularization=Tikhonov(),
                 lamb=0,
                 scale=1.0,
                 b=0.5,
                 tolerance=1e-5,
                 max_iterations=100):

        # calls super constructor with every parameter except the clippings
        Estimator.__init__(self,
                           regularization=regularization,
                           lamb=lamb,
                           scale=scale,
                           b=b,
                           tolerance=tolerance,
                           max_iterations=max_iterations)
        # creates two instances of the loss function with the two different
        # clippings
        self.loss_function_1 = loss_function(clipping=clipping_1)
        self.loss_function_2 = loss_function(clipping=clipping_2)

    # Returns the solution of the Tau-Estimator for the given inputs
    def estimate(self, a, y, initial_x=None):
        """
        This routine minimizes the objective function associated with the
        tau-estimator.
        For more information on the tau estimator see
        http://arxiv.org/abs/1606.00812

        This function is hard to minimize because it is non-convex. This
        means that it has several local minimums; depending on
        the initial x that is used, the algorithm ends up in a different
        local minimum.

        This algorithm takes the 'brute force' approach: it tries many
        different initial solutions, and picks the
        minimum with smallest value. The output of this estimation is the
        best minimum found.

        :param a: MxN matrix A in the y=Ax equation
        :type a: numpy.ndarray
        :param y: M vector y in the y=Ax equation
        :type y: numpy.ndarray
        :param initial_x: N vector of an initial solution
        :type initial_x: numpy.ndarray
        :return x_hat, tscalesquare: best estimation of the N vector x in the\
         y=Ax equation and value of the tau scale
        :rtype: Tuple[numpy.ndarray, numpy.float64]

        :Example:

        >>> import numpy as np
        >>> import linvpy as lp

        >>> a = np.matrix([[1, 2], [3, 4], [5, 6]])
        >>> y = np.array([1, 2, 3])

        >>> tau = lp.TauEstimator()
        >>> tau.estimate(a,y)
        (array([  1.45956448e-16,   5.00000000e-01]), 1.9242827743815571)

        >>> tau_ = lp.TauEstimator(loss_function=lp.Cauchy, clipping_1=2.23,\
         clipping_2=0.7, regularization=lp.Lasso(), lamb=2)
        >>> initial_solution = np.array([1, 2])
        >>> tau_.estimate(a, y, initial_x=initial_solution)
        (array([ 0.        ,  0.26464687]), 0.99364206111683273)
        """

        # ensures numpy types
        a = np.matrix(a)
        y = np.matrix(y)

        # If no initial solution is given, we create it as a vector of ones
        # Otherwise we use the one given
        if initial_x is None:
            x_hat = np.ones(a.shape[1])
        else:
            x_hat = initial_x

        # Computes the residual y - A * initial_x
        residuals = np.array(y.reshape(-1, 1) - np.dot(a, x_hat))

        # Estimates the scale using the residuals
        self.scale = np.median(np.abs(residuals)) / 0.6745

        # If the scale == 0 this means we have a good enough solution so we
        # return the current x_hat
        if self.scale == 0.0:
            return x_hat

        # x_hat = solution of the Tau version of irls
        x_hat = self.irls(a, y, x_hat)

        # residuals = y - A * x_hat
        residuals = y - a * x_hat.reshape(-1, 1)

        # tscalesquare = value of the objective function associated with
        # this x_hat
        tscalesquare = self.tau_scale(residuals)

        # we return the best solution we found, with the value of the objective
        # function associated with this x_hat
        return x_hat, tscalesquare

    def fast_estimate(self, a, y, initial_x=None, initial_iter=5):
        """
        Fast version of the basic tau algorithm. To save some computational
        cost, this algorithm exploits the speed of convergence of the basic
        algorithm. It has two steps: in the first one, for every initial
        solution, it only performs initialiter iterations. It keeps value of
        the objective function. In the second step, it compares the value of
        all the objective functions, and it select the nmin smaller ones. It
        iterates them until convergence. Finally, the algorithm select the x
        that produces the smallest objective function. For more details see
        http://arxiv.org/abs/1606.00812

        :param a: MxN matrix A in the y=Ax equation
        :type a: numpy.ndarray
        :param y: M vector y in the y=Ax equation
        :type y: numpy.ndarray
        :param initial_x: N vector of an initial solution
        :type initial_x: numpy.ndarray
        :param initial_iter: number of iterations to be performed
        :type initial_iter: integer
        :return x_hat, tscalesquare: best estimation of the N vector x in the\
         y=Ax equation and value of the tau scale
        :rtype: Tuple[numpy.ndarray, numpy.float64]

        :Example:

        >>> import numpy as np
        >>> import linvpy as lp

        >>> a = np.matrix([[1, 2], [3, 4], [5, 6]])
        >>> y = np.array([1, 2, 3])

        >>> tau = lp.TauEstimator()
        >>> tau.fast_estimate(a,y)
        (array([  1.45956448e-16,   5.00000000e-01]), 1.9242827743815571)

        >>> tau_ = lp.TauEstimator(loss_function=lp.Cauchy, clipping_1=2.23,\
         clipping_2=0.7, regularization=lp.Lasso(), lamb=2)
        >>> initial_solution = np.array([1, 2])
        >>> tau_.fast_estimate(a, y, initial_x=initial_solution,\
         initial_iter=6)
        (array([ 0.        ,  0.26464805]), 0.99363748192798829)
        """

        # stores the value of the default max iterations
        default_iter = self.max_iterations

        # sets the max iterations to the one given
        self.max_iterations = initial_iter

        # calls the basic estimate with a low max_iter to have a quick first
        #  solution
        temp_xhat, temp_tscale = self.estimate(a, y, initial_x=initial_x)

        # resets the number of iterations to the default one
        self.max_iterations = default_iter

        # calls the basic estimate again with the previous solution
        x_hat, tscalesquare = self.estimate(a, y, initial_x=temp_xhat)

        return x_hat, tscalesquare

    def tau_weights(self, x):
        # ensures numpy matrix type (x is a vector of size 1,n)
        x = np.matrix(x)

        # To avoid dividing by zero, if the sum is 0 it returns an array of
        # zeros
        if np.sum(self.loss_function_1.psi(x)) == 0:
            return np.zeros(x.shape[1])
        else:
            # Returns sum(2 * rho2 - psi2 * x) / sum(psi1 * x)
            return np.sum(
                2.0 * self.loss_function_2.rho(x) -
                np.multiply(self.loss_function_2.psi(x), x)) / \
                   np.sum(
                       np.multiply(self.loss_function_1.psi(x), x)
                   )

    # Score function for the tau estimator
    def score_function(self, x):
        # Computes the Tau weights
        tau_weights = self.tau_weights(x)

        # Score = tau_weights(x) * psi1(x) + psi2(x)
        # This sign(x) * abs() enforces that the function must be odd
        return np.sign(x) * abs(tau_weights * self.loss_function_1.psi(
            x) + self.loss_function_2.psi(x))

    def m_scale(self, x):
        # ensures array type
        x = np.array(x)

        # initial MAD estimation of the scale
        s = np.median(np.abs(x)) / 0.6745

        # If s==0 we have a good enough solution and return 0
        if s == 0:
            return 0.0

        rho_old = np.mean(
            self.loss_function_1.rho(x / s)
        ) - self.b
        k = 0

        while np.abs(rho_old) > self.tolerance and k < self.max_iterations:

            # If s==0 or the mean==0 we have a good enough solution and
            # return 0
            if (s == 0.0) or np.mean(self.loss_function_1.psi(
                            x / s) * x / s) == 0:
                return 0.0

            delta = rho_old / np.mean(
                self.loss_function_1.psi(x / s) * x / s) / s

            isqu = 1
            ok = 0
            while isqu < 30 and ok != 1:

                # If s==0 or s+delta we have a good enough solution and
                # return 0
                if (s == 0.0) or (s + delta == 0):
                    return 0.0

                rho_new = np.mean(
                    self.loss_function_1.rho(
                        x / (s + delta)
                    )
                ) - self.b

                if np.abs(rho_new) < np.abs(rho_old):
                    s = s + delta
                    ok = 1
                else:
                    delta /= 2
                    isqu += 1
                rho_old = rho_new
                k += 1

        return np.abs(s)

    # Computes the scale for the Tau-Estimator
    def tau_scale(self, x):
        mscale = self.m_scale(x)
        m = x.shape[0]

        if mscale == 0:
            tscale = 0.0
        else:
            tscale = mscale ** 2 * (1 / m) * \
                     np.sum(self.loss_function_2.rho(x / mscale))
        return tscale


# Use of doctest, a tool that verifies that the code examples outputs match
# the code examples inputs in the docstrings
if __name__ == "__main__":
    import doctest

    doctest.testmod()
