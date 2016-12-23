.. linvpy documentation master file, created by sphinx-quickstart on Thu Dec  8 09:50:43 2016.
.. You can adapt this file completely to your liking, but it should at least contain the root `toctree` directive.

.. image:: images/EPFL_logo.png
   :align: left
   :width: 20 %
.. image:: images/lcav_logo_.png
   :align: right
   :width: 20 %


|
|
|
|


Welcome to linvpy's documentation !
===================================

LinvPy is a Python package designed to solve linear inverse problems of the
form :

.. math:: y = Ax + n

where :math:`y` is a vector of measured values, :math:`A` is a known matrix,
:math:`x` is an unknown input vector and :math:`n` is noise.

The goal is to find :math:`x`, or at least the best possible estimation; if
the matrix :math:`A` is invertible, the solution is easy to find by
multiplying by the inverse, if not, we need to use regression techniques
such as least squares method to find :math:`x`. The first motivation for
this project is that Marta Martinez-Camara, PhD student in Communications
Systems at EPFL (Switzerland) designed some new algorithms for solving linear
inverse problems. LinvPy is a Python implementation of these algorithms,
which may not be available anywhere else than here. LinvPy also contains
several other known functions such as loss functions regularization
functions or M-estimators.

Source code is on GitHub : https://github.com/LCAV/linvpy.

Get it
======

LinvPy is available from PyPi and Python 2.7 compatible. If you have pip already installed, simply run : ::

    $ sudo pip2 install --ignore-installed --upgrade linvpy

If you don't have pip installed, run : ::

    $ sudo easy_install pip
    $ sudo pip2 install --ignore-installed --upgrade linvpy
    
Quick start
===========
To solve :math:`y=Ax` with outliers knowing :math:`y, A` : ::

    import numpy as np
    import linvpy as lp

    a = np.matrix([[1, 2], [3, 4], [5, 6]])
    y = np.array([1, 2, 3])

You can create a tau estimator object with the default parameters : ::

    # Using the Tau-estimator :
    tau = lp.TauEstimator()
    tau.estimate(a,y)
    # returns : (array([  1.45956448e-16,   5.00000000e-01]), 1.9242827743815571)
    # where array([  1.45956448e-16,   5.00000000e-01]) is the best x to solve y=Ax
    # and 1.9242827743815571 is the value of the tau scale for this x

Or an M estimator : ::

    # Using the M-estimator :
    m = lp.MEstimator()
    m.estimate(a,y)
    # returns [ -2.95552481e-16   5.00000000e-01], the best x to solve y=Ax

You can easily choose the loss function you want to use when you create the object : ::
    
    # By default both estimators use the Huber loss function, but you can use any of Huber, Cauchy, Bisquare or Optimal (all described in the doc below) :
    tau = lp.TauEstimator(loss_function=lp.Cauchy)
    tau.estimate(a,y)

And the rest of the parameters: ::

    # or you can give one, two, three... or all parameters :
    tau = lp.TauEstimator(
        loss_function=lp.Optimal,
        clipping_1=0.6,
        clipping_2=1.5,
        lamb=3,
        scale=1.5,
        b=0.7,
        tolerance=1e4, )
    tau.estimate(a,y)
    
    
Or you can change the parameters later: ::

    # to change the clipping or any other parameter of the estimator :
    tau.loss_function_1.clipping = 0.7
    tau.tolerance = 1e3
    m.lamb = 3

You can also choose a particular initial solution for the irls algorithm. To get the solution you run the method 'estimate' with your data a and y, and initial solution x_0 if any (this is not necessary): ::

    # running with an initial solution :
    x = np.array([5, 6])
    x_tau_estimate = tau.estimate(a,y, initial_x=x)
    m_tau_estimate = m.estimate(a,y, initial_x=x)




.. index::

Module contents
===============

.. automodule:: linvpy

.. rubric:: Estimators
.. autosummary::
   :nosignatures:

   MEstimator
   TauEstimator

.. rubric:: Loss Functions
.. autosummary::
   :nosignatures:

   Bisquare
   Cauchy
   Huber
   Optimal

.. rubric:: Regularization Functions
.. autosummary::
   :nosignatures:

   Lasso
   Tikhonov
   

Using custom loss functions
===========================

To use a custom loss function :
1) copy paste this code into your python file
2) change the name "CustomLoss" with the name of your loss function
3) change the two "0.7" with the value of your default clipping
4) define your rho function in the unit_rho definition
5) define your psi function as the derivative of the rho function in unit_psi
6) create your own tau estimator by passing your loss function name to it ::

# Define your own loss function
class CustomLoss(lp.LossFunction):

    # Set your custom clipping
    def __init__(self, clipping=0.7):
        lp.LossFunction.__init__(self, clipping)
        if clipping is None:
            self.clipping = 0.7

    # Define your rho function : you can copy paste this and just change what's
    # inside the unit_rho
    def rho(self, array):
        # rho function of your loss function on ONE single element
        def unit_rho(element):
            # Simply return clipping * element for example
            return element + self.clipping
        # Vectorize the function
        vfunc = np.vectorize(unit_rho)
        return vfunc(array)

    # Define your psi function as the derivative of the rho function : you can
    # copy paste this and just change what's inside the unit_rho
    def psi(self, array):
        # rho function of your loss function on ONE single element
        def unit_psi(element):
            # Simply return the clipping for example
            return 1
        # Vectorize the function
        vfunc = np.vectorize(unit_psi)
        return vfunc(array)

a = np.matrix([[1, 2], [3, 4], [5, 6]])
y = np.array([1, 2, 3])

custom_tau = lp.TauEstimator(loss_function=CustomLoss)
print custom_tau.estimate(a,y)

Using custom regularization functions
=====================================

To use a custom regularization function :
1) copy paste this code into your python file
2) change the name CustomRegularization with the name of your function
3) define the regularization function in the definition of regularize
4) create your custom tau by passing an instance of your regularization with "()"

# Define your own regularization
class CustomRegularization(lp.Regularization):
    pass
    # Define your regularization function here
    def regularize(self, a, y, lamb=0):
        return np.ones(a.shape[1])

a = np.matrix([[1, 2], [3, 4], [5, 6]])
y = np.array([1, 2, 3])

# Create your custom tau estimator with custom regularization function
# Pay attenation to pass the loss function as a REFERENCE (without the "()"
# after the name, and the regularization as an OBJECT, i.e. with the "()").
custom_tau = lp.TauEstimator(regularization=CustomRegularization())
print custom_tau.estimate(a,y)

Tutorial
========
Why do we need robust estimators?
---------------------------------
The nature of the errors that appear in a problem may pose a significant challenge. This is quite an old problem, and it was already mentioned in the first publications about least squares, more than two centuries ago. Legendre wrote in 1805 

 .. epigraph::
    If among these errors are some which appear too large to be admissible, then those   observations which produced these errors will be rejected, as coming from too faulty   experiments, and the unknowns will be determined by means of the other observations, which will then give much smaller errors.

 .. figure:: images/outlier_effect.png
    :scale: 70 %

Contribute
==========

If you want to contribute to this project, feel free to fork our GitHub main repository repository : https://github.com/LCAV/linvpy. Then, submit a 'pull request'. Please follow this workflow, step by step:

1. Fork the project repository: click on the 'Fork' button near the top of the page. This creates a copy of the repository in your GitHub account.

2. Clone this copy of the repository in your local disk.

3. Create a branch to develop the new feature : ::

    $ git checkout -b new_feature

 Work in this branch, never in the master branch.

4. To upload your changes to the repository : ::

    $ git add modified_files
    $ git commit -m "what did you implement in this commit"
    $ git push origin new_feature

When your are done, go to the webpage of the main repository, and click 'Pull request' to send your changes for review.


Documentation
=============

.. module::
.. automodule:: linvpy
   :members:
   :exclude-members: m_scale, score_function, tau_weights, tau_scale, m_weights, Estimator, LossFunction, Regularization
   :undoc-members:
   :show-inheritance:
   

Indices and tables
==================

:ref:`genindex`
:ref:`modindex`
:ref:`search`