import numpy as np


def rho_inverse(u):
    '''
    inverse function of rho
    :param u: scalar input for the function
    :return: another scalar
    '''

    f = np.sqrt(1 - (1 - u) ** (1 / 3))
    return f


def mscale(x, normz=1, b=0.5, tolerance=1e-5):
    '''
    Implementation of the m-scale from Michael's code
    :param x: data to compute the m scale
    :param normz: (optional) if > o, normalized sigma for consistency
    :param b: b in Equation 7 of our paper
    :param tolerance: error tolerance
    :return: m scale of x
    '''

    # get dimensions of data
    n = len (x)

    # sorting data
    y = np.sort(np.abs(x))

    # ask why do we do this
    n1 = np.floor(n * (1 - b))
    n2 = np.ceil(n * (1 - b) / (1 - b / 2))

    # bounds for the data?
    qq = [y[n1 - 1], y[n2 - 1]]

    # rho inverse for b\2
    binverse = rho_inverse(b / 2)

    # initial interval where sigma is
    sigma_initial = [qq[0], qq[1] / binverse]

    # relative or absolute tolerance, for sigma> or < 1
    if qq[0] >= 1:
      tol = tolerance
    else:
      tol = tolerance * qq[0]

    # compute the sigma for this iteration
    if np.mean(x[x == 0] > (1 - b)):
      sigma_m = 0
    else:
      print 'find roots of a function'

    # TBD


def regularized_s(x, y, nlambdas, cvtype, nkeep, niter, verbose):
    '''
    This function calculates the ridge regression estimate.
    It is the traslation of the matlab code by Michael's student.

    Dependencies: PeYoRid method, mscale
    :param x: regression matrix (N x P)
    :param y: measurement vector (N x 1)
    :param nlambdas: number of lambdas values we use
    :param cvtype: type of the cross-validation (write options)
    :param nkeep: number of candidates to be kept for full iteration
    :param niter: maximum number of iterations for S-ridge calculation
    :return:
    '''


    print 'here goes the translation of matlab code'


def main():
    fakedata = np.random.rand(10)
    sigma = mscale(fakedata)


if __name__ == '__main__':
    # To launch manually in console:
    # python sridge.py
    main()
