import os
from setuptools import setup

setup(
    name = "A Python package for linear inverse problems",
    version = "0.0.1",
    author = "Guillaume Beaud, Marta Martinez-Camara",
    author_email = "beaudguillaume@gmail.com",
    description = ("Package to solve some linear inverse problems."),
    license = "BSD",
    keywords = "linear inverse",
    url = "http://packages.python.org/an_example_pypi_project",
    packages=['linear_inverse'],
    #long_description=read('README'),
    classifiers=[
        "Development Status :: 1 - Planning"
    ],
    install_requires=[
          'numpy', 'scikit-learn', 'scipy'
      ]
)