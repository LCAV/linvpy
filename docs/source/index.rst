.. linvpy documentation master file, created by sphinx-quickstart on Thu Dec  8 09:50:43 2016.
.. You can adapt this file completely to your liking, but it should at least contain the root `toctree` directive.

.. image:: images/lcav_logo.png
   :width: 20 %
.. image:: images/EPFL_logo.png
   :width: 20 %

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

LinvPy is available from PyPi. If you have pip already installed, simply run : ::

    $ sudo pip install linvpy

If you don't have pip installed, run : ::

    $ sudo easy_install pip
    $ sudo pip install linvpy

To upgrade linvpy to the latest version : ::

    $ sudo pip install --upgrade linvpy
    
Quick start
===========
The main functions you may want to use from this package are the tau-estimator and the M-estimator (documentation below), which estimate the best value of x to solve :math:`y=Ax` : ::

    import numpy as np
    import linvpy as lp

    a = np.matrix([[1, 2], [3, 4], [5, 6]])
    y = np.array([1, 2, 3])

    # create an instance of tau estimator, don't need to give any parameter
    tau = lp.TauEstimator()

    tau.estimate(a,y)
    # returns : (array([  1.45956448e-16,   5.00000000e-01]), 1.9242827743815571)
    # where array([  1.45956448e-16,   5.00000000e-01]) is the best x to solve y=Ax
    # and 1.9242827743815571 is the value of the tau scale for this x

    # By default it uses the Huber loss function, but you can use any of Huber, Cauchy, Bisquare or Optimal (all described in the doc below) :
    tau_ = lp.TauEstimator(loss_function=lp.Cauchy)
    tau_.estimate(a,y)

    # or you can give one, two, three... or all parameters :
    tau__ = lp.TauEstimator(
        loss_function=lp.Optimal,
        clipping_1=0.6,
        clipping_2=1.5,
        lamb=3,
        scale=1.5,
        b=0.7,
        tolerance=1e4, )
    tau__.estimate(a,y)

    # creates an instance of M-estimator :
    my_m = lp.MEstimator()
    my_m.estimate(a,y)

    # to change the clipping or any other parameter of the estimator :
    tau.loss_function_1.clipping = 0.7
    tau.tolerance = 1e3

    # running with an initial solution :
    x = np.array([5, 6])
    tau_.estimate(a,y, initial_x=x)




Documentation
=============
.. index::

Module contents
---------------

.. module::
.. automodule:: linvpy
   :members:
   :exclude-members: m_scale, score_function, tau_weights, tau_scale, m_weights, Estimator, LossFunction, Regularization
   :undoc-members:
   :show-inheritance:

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


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

