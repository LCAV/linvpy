.. linvpy documentation master file, created by
   sphinx-quickstart on Tue Mar 08 09:33:29 2016.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.


Welcome to LinvPy !
===================

LinvPy is a Python package designed for solving linear inverse problems of the 
form :math:`y = Ax + n`, where y is a vector of measured values, A a known matrix, x an unknown input vector and n is noise. The goal is to find x, or at least the best possible estimation; if the matrix A is invertible, the solution is easy to find by multiplying by the inverse, if not, we need to use regression techniques such as least squares method to find x.

The first motivation for this project is that Marta Martinez-Camara, PhD student in Communications Systems at EPFL (Switzerland) designed some new algorithms for solving linear inverse problems. LinvPy is a Python implementation of these algorithms, which may not be available anywhere else than here. LinvPy also contains several other known and available techniques such as least squares regression, regularization functions, or M-estimators.

Get it
======

LinvPy is available from PyPi. If you have pip already installed, simply run : ::

	$ sudo pip install linvpy

If you don't have pip installed, run : ::

	$ sudo easy_install pip
	$ sudo pip install linvpy

To upgrade linvpy to the latest version : ::

	$ sudo pip install --upgrade linvpy


Documentation
=============

.. module::
.. automodule:: linvpy
.. autofunction:: least_squares
.. autofunction:: tikhonov_regularization
.. autofunction:: rho_huber
.. autofunction:: psi_huber
.. autofunction:: rho_bisquare
.. autofunction:: psi_bisquare
.. autofunction:: rho_cauchy
.. autofunction:: psi_cauchy
.. autofunction:: rho_optimal
.. autofunction:: weights
.. autofunction:: irls
.. autofunction:: weights
.. autofunction:: irls

Contribute
=======

If you want to contribute to this project, feel free to fork our GitHub main repository repository : https://github.com/GuillaumeBeaud/linvpy. Then, submit a 'pull request'. Please follow this workflow, step by step:

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

