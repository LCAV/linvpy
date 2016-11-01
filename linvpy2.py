__author__ = 'GuillaumeBeaud'

import numpy as np
import math
import matplotlib.pyplot as plt


# Abstract class for loss functions so they share the same interface and all have the rho, psi, weights functions
class LossFunction:
    def __init__(self, clipping=None):  # Constructor of the class
        if clipping is not None:
            assert clipping > 0  # verifies clipping is >0 if it is not None
            self.clipping = clipping

    def rho(self, element):  # Abstract method, defined by convention only
        raise NotImplementedError("Subclass must implement abstract method")

    def psi(self, element):  # Abstract method, defined by convention only
        raise NotImplementedError("Subclass must implement abstract method")

    # classic weights for m-estimator
    def m_weights(self, matrix):
        def unit_operation(element):  # operation on a single element, which is then vectorized
            if element == 0:
                return 0
            else:
                return self.psi(element) / element

        vfunc = np.vectorize(unit_operation)  # unit_operation is vectorized to operate on a matrix
        return vfunc(matrix)  # returns the matrix with the unit_operation executed on each element

    # Plots the rho, psi and weights on the given interval
    def plot(self, interval):
        plt.plot([self.rho(i) for i in range(-interval, interval)], label=self.__class__.__name__ + ' rho')
        plt.plot([self.psi(i) for i in range(-interval, interval)], label=self.__class__.__name__ + ' psi')
        plt.plot([self.m_weights(i) for i in range(-interval, interval)], label=self.__class__.__name__ + ' weights')
        plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=2, mode="expand", borderaxespad=0.)
        plt.show()


class Huber(LossFunction):
    def __init__(self, clipping=1.345):
        LossFunction.__init__(self, clipping)
        if clipping is None:
            self.clipping = 1.345

    def rho(self, element):
        if abs(element) <= self.clipping:
            return math.pow(element, 2) / 2.0
        else:
            return self.clipping * abs(element) * self.clipping / 2.0

    def psi(self, element):
        if abs(element) >= self.clipping:
            return self.clipping * np.sign(element)
        else:
            return element


class Bisquare(LossFunction):
    def __init__(self, clipping=4.685):
        LossFunction.__init__(self, clipping)
        if clipping is None:
            self.clipping = 4.685

    def rho(self, element):
        if abs(element) <= self.clipping:
            return ((self.clipping ** 2.0) / 6.0) * (1 - (1 - (element / self.clipping) ** 2) ** 3)
        else:
            return (self.clipping ** 2) / 6.0

    def psi(self, element):
        if abs(element) <= self.clipping:
            return element * ((1 - (element / self.clipping) ** 2) ** 2)
        else:
            return 0.0


class Cauchy(LossFunction):
    def __init__(self, clipping=2.3849):
        LossFunction.__init__(self, clipping)
        if clipping is None:
            self.clipping = 2.3849

    def rho(self, element):
        return ((self.clipping ** 2) / 2) * math.log(1 + (element / self.clipping) ** 2)

    def psi(self, element):
        return element / (1 + (element / self.clipping) ** 2)


class Optimal(LossFunction):
    def __init__(self, clipping=3.270):
        LossFunction.__init__(self, clipping)
        if clipping is None:
            self.clipping = 3.270

    def rho(self, element):
        if abs(element / self.clipping) <= 2.0 / 3.0:
            return 1.38 * (element / self.clipping) ** 2
        elif abs(element / self.clipping) <= 1.0:
            return 0.55 - (2.69 * (element / self.clipping) ** 2) + (
                10.76 * (element / self.clipping) ** 4) - (
                       11.66 * (element / self.clipping) ** 6) + (
                       4.04 * (element / self.clipping) ** 8)
        elif abs(element / self.clipping) > 1:
            return 1.0

    def psi(self, element):
        if abs(element / self.clipping) <= 2.0 / 3.0:
            return 2 * 1.38 * (element / self.clipping ** 2)
        elif abs(element / self.clipping) <= 1.0:
            return (- 2 * 2.69 * (element / self.clipping ** 2)) + (
                4 * 10.76 * (element ** 3 / self.clipping ** 4)) - (
                       6 * 11.66 * (element ** 5 / self.clipping ** 6)) + (
                       8 * 4.04 * (element ** 7 / self.clipping ** 8))
        elif abs(element / self.clipping) > 1:
            return 0


# Abstract class for regularization functions so they share the same interface
class Regularization:
    pass  # no constructor needed. This class is used as an interface for all the regularization functions

    def regularize(self, A, y):  # Abstract method, defined by convention only
        raise NotImplementedError("Subclass must implement abstract method")


class Tikhonov(Regularization):
    pass

    # returns th tikhonov regularization from A,y,lambda
    def regularize(self, A, y, lamb=0):
        assert lamb >= 0
        y = np.squeeze(np.asarray(y))  # flattens y into a vector
        if lamb == 0:  # if lambda == 0 we simply return the least squares solution
            return np.linalg.lstsq(A, y)[0].reshape(-1)
        else:
            identity_matrix = np.identity(A.shape[1])
            # output = (A' A + lambda^2 I)^-1 A' y
            xhat = np.dot(
                np.dot(
                    np.linalg.inv(
                        np.add(
                            np.dot(A.T, A),
                            np.dot(lamb ** 2, identity_matrix)
                        ),
                    ), A.T
                ), y)
            return np.squeeze(np.asarray(xhat))  # flattens result into an array


# super class of the M and Tau Estimators. All values are default so you can simply create one
# with my_estimator = MEstimator() and then my_estimator.estimate(A,y) which gives the answer
class MEstimator():
    def __init__(self,
                 loss_function=Huber,
                 clipping=None,
                 regularization=Tikhonov(),
                 lamb=0,
                 scale=1.0,
                 initial_x=None,
                 b=0.5,
                 tolerance=1e-5,
                 max_iterations=100):
        assert scale != 0
        self.loss_function = loss_function(clipping=clipping)
        self.regularization = regularization
        self.lamb = lamb
        self.scale = scale
        self.initial_x = initial_x
        self.b = b
        self.tolerance = tolerance
        self.max_iterations = max_iterations

    def irls(self, A, y):

        # if an initial value for x is specified, use it, otherwise generate a
        # vector of ones
        if self.initial_x is not None:
            vector_x = self.initial_x
        else:
            # Generates a ones vector_x with length = A.columns
            vector_x = np.ones(A.shape[1])
            initial_x = np.ones(A.shape[1])

            # Ensures numpy types
            A = np.matrix(A)
            y = y.reshape(-1, 1)

            # Residuals = y - Ax, difference between measured values and model
            residuals = y - np.dot(A, initial_x).reshape(-1, 1)

        for i in range(1, self.max_iterations):

            # normalize residuals :rhat = ((y - Ax)/ self.scale)
            rhat = np.array(residuals / self.scale).flatten()

            # weights_vector = weights of rhat according to the loss function
            weights_vector = self.loss_function.m_weights(rhat)

            # Makes a diagonal matrix with values of w(y-Ax)
            # np.squeeze(np.asarray()) is there to flatten the matrix into a vector
            weights_matrix = np.diag(
                np.squeeze(
                    np.asarray(weights_vector)
                )
            )

            # Square root of the weights matrix, sqwm = W^1/2
            sqwm = np.sqrt(weights_matrix)

            # A_weighted = W^1/2 A
            a_weighted = np.dot(sqwm, A)

            # y_weighted = diagonal of W^1/2 y
            y_weighted = np.dot(sqwm, y)

            # vector_x_new is there to keep the previous value to compare
            vector_x_new = self.regularization.regularize(A, y, self.lamb)

            # Normalized distance between previous and current iteration
            xdis = np.linalg.norm(vector_x - vector_x_new)

            # New residuals
            residuals = y.reshape(-1) - np.dot(A, vector_x_new).reshape(-1)

            # Divided by the specified optional self.scale, otherwise self.scale = 1
            vector_x = vector_x_new

            # if the difference between iteration n and iteration n+1 is smaller
            # than self.tolerance, return vector_x
            if (xdis < self.tolerance):
                return vector_x

        return vector_x

    def estimate(self, A, y):  # Abstract method so all estimators have a estimator.estimate function
        return self.irls(A, y)
