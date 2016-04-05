.. linvpy documentation master file, created by
   sphinx-quickstart on Tue Mar 08 09:33:29 2016.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.


Welcome to LinvPy !
===================

LinvPy is a Python package designed for solving linear inverse problems of the 
form :math:`y = Ax + n`, where y is a vector of measured values, A a known matrix, x an unknown input vector and n is noise. The goal is to find x, or at least the best possible estimation; if the matrix A is invertible, the solution is easy to find by multiplying by the inverse, if not, we need to use regression techniques such as least squares method to find x.

The first motivation for this project is that Marta Martinez-Camara, PhD student in Communications Systems at EPFL (Switzerland) desgined some new algorithms for solving linear inverse problems, and this package is a Python implementation of these algorithms, which may not be available anywhere else than here. LinvPy also contains several other known and available techniques such as least squares regression, regularization functions, loss functions or M-estimators which you can also find in the famous Numpy or Scipy packages.

Get it
======

LinvPy is available from PyPi. If you have pip already installed, simply run : ::

	$ pip install linvpy

If you don't have pip installed, run : ::

	$ sudo easy_install pip
	$ pip install linvpy


Documentation
=============

.. module:: regression
.. autofunction:: least_squares
.. autofunction:: least_squares_gradient
.. autofunction:: tikhonov_regularization
.. autofunction:: huber_loss
.. autofunction:: weight_function
.. autofunction:: iteratively_reweighted_least_squares
.. autofunction:: phi


Develop
=======

If you want to contribute to this project, feel free to create a new branch on our GitHub repository : https://github.com/GuillaumeBeaud/linvpy

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

