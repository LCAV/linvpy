.. linvpy documentation master file, created by sphinx-quickstart on Thu Dec  8 09:50:43 2016.
.. You can adapt this file completely to your liking, but it should at least contain the root `toctree` directive.

Welcome to linvpy's documentation !
===================================

LinvPy is a Python package designed to solve linear inverse problems of the
form :

.. math:: y = Ax + n

where :math:`y` is a vector of measured values, :math:`A` a known matrix,
:math:`x` an unknown input vector and :math:`n` is noise.

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

Documentation
=============


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
----------------------------------

 .. figure:: images/outlier_effect.png

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

