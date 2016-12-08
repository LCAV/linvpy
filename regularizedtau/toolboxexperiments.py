"""
===================================================================
EXPERIMENTS
This toolbox contains all the experiments necessary for the paper.
Marta Martinez-Camara, EPFL
===================================================================
"""
from __future__ import division  # otherwise we get just an int as output

import toolboxutilities as util
import numpy as np
import sys
import matplotlib.pyplot as plt
import pickle
import toolboxinverse as inv
import linvpy as ln
import os
import gflags
import scipy.io
from mpl_toolkits.mplot3d import Axes3D

# initialize gflags
FLAGS = gflags.FLAGS

gflags.DEFINE_string(
    'figure',
    'bias',
    'What figure do you want to get',
)

# directory to store the figures of experimental results
FIGURES_DIR = os.path.join(
    os.path.dirname(__file__),
    "figures"
)

# directory to store the data of experimental results
DATA_DIR = os.path.join(
    os.path.dirname(__file__),
    "results_data"
)


# -------------------------------------------------------------------
# Sensitivity curve function
# -------------------------------------------------------------------
def sensitivitycurve(estimator, lossfunction, regularization, yrange, arange, nmeasurements, points):

    x = 1.5  # fixed source
    a = util.getmatrix(1, 'random', nmeasurements)  # get the sensing matrix
    y = util.getmeasurements(a, x, 'gaussian')
    if estimator == 'tau' and regularization == 'none':
        # xhat, shat = inv.fasttau(y, a, lossfunction, (0.4, 1.09), 10, 3, 3)  # solution without outliers
        xhat, shat = ln.basictau(
            a,
            y,
            'optimal',
            [0.4, 1.09],
            ninitialx=30,
            maxiter=100,
            nbest=1,
            lamb=0
        )
    elif estimator == 'tau' and regularization == 'l2':
        # xhat, shat = inv.basictauridge(y, a, lossfunction, (0.4, 1.09), 20, 0.1)  # solution without outliers
        xhat, shat = ln.basictau(
            a,
            y,
            'optimal',
            [0.4, 1.09],
            ninitialx=30,
            maxiter=100,
            nbest=1,
            regularization=ln.tikhonov_regularization,
            lamb=0.1
        )
    elif estimator == 'tau' and regularization == 'l1':
        # xhat, shat = inv.fasttaulasso(y, a, lossfunction, (0.4, 1.09), 30, 0.1)  # solution without outliers
        xhat, shat = ln.basictau(
            a,
            y,
            'optimal',
            [0.4, 1.09],
            ninitialx=30,
            maxiter=100,
            nbest=1,
            regularization=ln.lasso_regularization,
            lamb=0.1
        )

    elif estimator == 'M' and regularization == 'none':
        xhat = inv.mestimator(y, a, lossfunction, 1.345, 1)  # solution without outliers
    else:
        sys.exit('I do not know the estimator % s' % estimator)  # die gracefully
    y0 = np.linspace(-yrange, yrange, points)
    a0 = np.linspace(-arange, arange, points)
    sc = np.zeros((points, points))
    for i in np.arange(points):
        for j in np.arange(points):
            yout = np.insert(y, nmeasurements, y0[i])
            yout = np.expand_dims(yout, axis=1)
            aout = np.insert(a, nmeasurements, a0[j])
            aout = np.expand_dims(aout, axis=1)
            if estimator == 'tau' and regularization == 'none':
                # xout, sout = inv.fasttau(yout, aout, lossfunction, (0.4, 1.09), 10, 3, 3)  # solution with one outlier
                xout, sout = ln.basictau(
                    aout,
                    yout,
                    'optimal',
                    [0.4, 1.09],
                    ninitialx=10,
                    maxiter=100,
                    nbest=1,
                    lamb=0
                )
                sc[i, j] = (xout - xhat) / (1 / (nmeasurements + 1))
            elif estimator == 'M' and regularization == 'none':
                xout = inv.mestimator(yout, aout, 'huber', 1.345, 1)  # solution with one outlier
                sc[i, j] = (xout - xhat) / (1 / (nmeasurements + 1))
            elif estimator == 'tau' and regularization == 'l2':
                xout, sout = inv.basictauridge(yout, aout, lossfunction, (0.4, 1.09), 20, 0.1)  # solution with one outlier
                sc[i, j] = (xout - xhat) / (1 / (nmeasurements + 1))
            elif estimator == 'tau' and regularization == 'l1':
                #xout, sout = inv.fasttaulasso(yout, aout, lossfunction, (0.4, 1.09), 30, 0.1)  # solution with one outlier
                xout, sout = ln.basictau(
                    aout,
                    yout,
                    'optimal',
                    [0.4, 1.09],
                    ninitialx=30,
                    maxiter=100,
                    nbest=1,
                    regularization=ln.lasso_regularization,
                    lamb=0.01
                )
                sc[i, j] = (xout - xhat) / (1 / (nmeasurements + 1))
            else:
                sys.exit('I do not know the estimator % s' % estimator)  # die gracefully

    x, y = np.mgrid[0:points, 0:points]

    name_file = 'sc_' + regularization + '.pkl'
    fl = os.path.join(DATA_DIR, name_file)
    f = open(fl, 'wb')
    pickle.dump(sc, f)
    f.close()

    f = sc.T  # transpose it, for correct visualization
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.plot_surface(x, y, f, rstride=1, cstride=1)
    plt.xlabel('a0')
    plt.ylabel('y0')

    name_file = 'sc_' + regularization + '.eps'
    fl = os.path.join(FIGURES_DIR, name_file)
    fig.savefig(fl, format='eps')

    return sc


# -------------------------------------------------------------------
# Bias depending on lambda
# -------------------------------------------------------------------
def bias(estimator, regularization, lrange, lstep, nrealizations):

    lmbds = np.arange(0, lrange, lstep)
    nlmbd = np.size(lmbds)
    avrg = np.zeros((nlmbd, 1))
    sourcesize = 1  # dimension of the source
    matrixtype = 'random'  # type of sensing matrix
    noisetype = 'gaussian'  # additive noise
    measurementsize = 1000  # number of measurements
    x = 1.5
    for idx, lbd in enumerate(lmbds):
        print 'current lambda = ', lbd
        k = 0  # counter
        while k < nrealizations:
            a = util.getmatrix(sourcesize, matrixtype, measurementsize)  # get the sensing matrix
            y = util.getmeasurements(a, x, noisetype)  # get the measurements
            if estimator == 'tau' and regularization == 'l2':
                # xhat, obj = inv.basictauridge(y, a, 'optimal', (0.4, 1), 10, lmbds[i])
                xhat, obj = ln.basictau(
                  a,
                  y,
                  'optimal',
                  [0.4, 1.09],
                  ninitialx=30,
                  maxiter=100,
                  nbest=1,
                  regularization=ln.tikhonov_regularization,
                  lamb=lbd
              )
            elif estimator == 'tau' and regularization == 'l1':
              # xhat, obj = inv.fasttaulasso(y, a, 'optimal', (0.4, 1), 10, lmbds[i])
              xhat, obj = ln.basictau(
                  a,
                  y,
                  'optimal',
                  [0.4, 1.09],
                  ninitialx=30,
                  maxiter=100,
                  nbest=1,
                  regularization=ln.lasso_regularization,
                  lamb=lbd
              )

            elif estimator == 'ls' and regularization == 'l2':
                xhat = inv.ridge(y, a, lbd)
            elif estimator == 'ls' and regularization == 'l1':
                xhat = inv.lasso(y, a, lbd)
            elif estimator == 'huber' and regularization == 'l2':
                xhat = inv.mridge(y, a, 'huber', 1.345, 1, lbd)
            else:
                sys.exit('I do not know the estimator % s' % estimator)  # die gracefully
            avrg[idx] = avrg[idx] + xhat
            k += 1
    avg = avrg / nrealizations
    bs = x - avg

    # store results
    name_file = 'bs' + '_' + regularization + '.pkl'
    fl = os.path.join(DATA_DIR, name_file)
    f = open(fl, 'wb')
    pickle.dump([bs, lmbds], f)
    f.close()

    # plot results
    fig = plt.figure()
    plt.plot(lmbds, bs)
    plt.xlabel('lambda')
    plt.ylabel('bias')

    # dump plot into file
    name_file = 'bs' + '_' + regularization + '.eps'
    fl = os.path.join(FIGURES_DIR, name_file)
    fig.savefig(fl, format='eps')
    # plt.show()
    return bs, lmbds

# -------------------------------------------------------------------
# Asymp. Variance depending on lambda. Numerical simulations
# -------------------------------------------------------------------


def asv(estimator, regularization, lrange, lstep, nrealizations):

    lmbds = np.arange(0, lrange, lstep)
    nlmbd = np.size(lmbds)
    estimates = np.zeros((nlmbd, nrealizations))
    sourcesize = 1  # dimension of the source
    matrixtype = 'random'  # type of sensing matrix
    noisetype = 'gaussian'  # additive noise
    measurementsize = 1000  # number of measurements
    x = 1.5
    for idx, lbd in enumerate(lmbds):
        print 'Current lambda =', lbd
        k = 0  # counter
        while k < nrealizations:
            a = util.getmatrix(sourcesize, matrixtype, measurementsize)  # get the sensing matrix
            y = util.getmeasurements(a, x, noisetype)  # get the measurements
            if estimator == 'tau' and regularization == 'l2':
                # xhat, obj = inv.basictauridge(y, a, 'optimal', (0.4, 1), 10, lmbds[i])
                xhat, obj = ln.basictau(
                    a,
                    y,
                    'optimal',
                    [0.4, 1],
                    ninitialx=10,
                    maxiter=100,
                    nbest=1,
                    regularization=ln.tikhonov_regularization,
                    lamb=lbd
                  )
            elif estimator == 'tau' and regularization == 'l1':
                #xhat, obj = inv.fasttaulasso(y, a, 'optimal', (0.4, 1), 10, lmbds[i])
                xhat, obj = ln.basictau(
                    a,
                    y,
                    'optimal',
                    [0.4, 1],
                    ninitialx=10,
                    maxiter=100,
                    nbest=1,
                    regularization=ln.lasso_regularization,
                    lamb=lbd
                )
            elif estimator == 'ls' and regularization == 'l2':
                xhat = inv.ridge(y, a, lbd)
            elif estimator == 'ls' and regularization == 'l1':
                xhat = inv.lasso(y, a, lbd)
            elif estimator == 'huber' and regularization == 'l2':
                xhat = inv.mridge(y, a, 'huber', 1.345, 1, lbd)
            else:
                sys.exit('I do not know the estimator % s' % estimator)  # die gracefully
            estimates[idx, k] = xhat
            k += 1
    vr = np.var(estimates, 1)
    av = measurementsize * vr

    avg = np.mean(estimates, 1)

    # store results
    name_file = 'asv_' + regularization + '.pkl'
    fl = os.path.join(DATA_DIR, name_file)
    f = open(fl, 'wb')
    pickle.dump([av, avg, lmbds], f)
    f.close()

    fig = plt.figure()
    plt.plot(lmbds, av)
    plt.xlabel('lambda')
    plt.ylabel('asv')

    name_file = 'asv_' + regularization + '.eps'
    fl = os.path.join(FIGURES_DIR, name_file)
    fig.savefig(fl, format='eps')
    # plt.show()

    return av, lmbds


# -------------------------------------------------------------------
# Experiment 1: non regularized case. Comparison LS, M, tau
# -------------------------------------------------------------------
def experimentone(nrealizations, outliers, measurementsize, sourcesize, source):

    sourcetype = 'random'  # kind of source we want
    matrixtype = 'random'  # type of sensing matrix
    noisetype = 'outliers'  # additive noise
    clippinghuber = 1.345  # clipping parameter for the huber function
    clippingopt = (0.4, 1.09)  # clipping parameters for the opt function in the tau estimator
    ninitialsolutions = 10  # how many initial solutions do we want in the tau estimator
    realscale = 1
    var = 1
    x = source
    # x = util.getsource(sourcetype, sourcesize)  # get the ground truth
    a = util.getmatrix(sourcesize, matrixtype, measurementsize)  # get the sensing matrix
    noutliers = outliers.size
    averrorls = np.zeros((noutliers, 1))  # store results for ls
    averrorm = np.zeros((noutliers, 1))  # store results for m
    averrormes = np.zeros((noutliers, 1))  # store results for m with an estimated scale
    averrortau = np.zeros((noutliers, 1))  # store results for tau
    k = 0
    while k < noutliers:
        r = 0
        while r < nrealizations:
            y = util.getmeasurements(a, x, noisetype, var, outliers[k])  # get the measurements
            # -------- ls solution
            xhat = inv.leastsquares(y, a)  # solving the problem with ls
            error = np.linalg.norm(x - xhat)
            averrorls[k] += error
            # -------- m estimated scale solution
            xpreliminary = xhat  # we take the ls to estimate a preliminary scale
            respreliminary = y - np.dot(a, xpreliminary)
            estimatedscale = np.median(np.abs(respreliminary)) / .6745  # robust mad estimator for the scale
            xhat = inv.mestimator(y, a, 'huber', clippinghuber, estimatedscale)  # solving the problem with m
            error = np.linalg.norm(x - xhat)
            averrormes[k] += error
            # -------- m real scale solution
            xhat = inv.mestimator(y, a, 'huber', clippinghuber, realscale)  # solving the problem with m
            error = np.linalg.norm(x - xhat)
            averrorm[k] += error
            # -------- tau solution
            # xhat, scale = inv.fasttau(y, a, 'optimal', clippingopt, ninitialsolutions)  # solving the problem with tau
            xhat, obj = ln.basictau(
                a,
                y,
                'optimal',
                clippingopt,
                ninitialx=ninitialsolutions,
                maxiter=100,
                nbest=1,
                lamb=0
            )
            error = np.linalg.norm(x - xhat)
            averrortau[k] += error
            r += 1  # update the number of realization
        k += 1  # updated the number of outlier proportion

    averrorls = averrorls / nrealizations  # compute average
    averrorm = averrorm / nrealizations
    averrormes = averrormes / nrealizations
    averrortau = averrortau / nrealizations

    # store results
    name_file = 'experiment_one.pkl'
    fl = os.path.join(DATA_DIR, name_file)
    f = open(fl, 'wb')
    pickle.dump([averrorls, averrorm, averrormes, averrortau], f)
    f.close()

    fig = plt.figure()
    plt.plot(outliers, averrorls, 'r--', label='ls')
    plt.plot(outliers, averrorm, 'bs--', label='m estimator')
    plt.plot(outliers, averrormes, 'g^-', label='m est. scale')
    plt.plot(outliers, averrortau, 'kd-', label='tau')
    plt.legend(loc=2)
    plt.xlabel('% outliers')
    plt.ylabel('error')

    name_file = 'experiment_one.eps'
    fl = os.path.join(FIGURES_DIR, name_file)
    fig.savefig(fl, format='eps')
    # plt.show()


# -------------------------------------------------------------------
# Experiment 2: l2 regularized case. Comparison LS, M, tau
# -------------------------------------------------------------------
def experimenttwo(nrealizations, outliers, measurementsize, sourcesize, source):

    matrixtype = 'illposed'  # type of sensing matrix
    conditionnumber = 1000  # condition number of the matrix that we want
    noisetype = 'outliers'  # additive noise
    clippinghuber = 1.345  # clipping parameter for the huber function
    clippingopt = (0.4, 1.09)  # clipping parameters for the opt function in the tau estimator
    ninitialsolutions = 50  # how many initial solutions do we want in the tau estimator
    realscale = 1
    var = 3
    x = source  # load stored source
    # x = util.getsource(sourcetype, sourcesize)  # get the ground truth
    a = util.getmatrix(sourcesize, matrixtype, measurementsize, conditionnumber)  # get the sensing matrix
    noutliers = outliers.size
    nlmbd = 6  # how many different lambdas are we trying in each case

    lmbdls = np.zeros((noutliers, nlmbd))  # every proportion of outliers need a different lambda
    lmbdls[0, :] = np.logspace(0, 3, nlmbd)  # lambdas for ls
    lmbdls[1, :] = np.logspace(7, 10, nlmbd)  # lambdas for ls
    lmbdls[2, :] = np.logspace(8, 11, nlmbd)  # lambdas for ls
    lmbdls[3, :] = np.logspace(8, 11, nlmbd)  # lambdas for ls
    lmbdls[4, :] = np.logspace(9, 11, nlmbd)  # lambdas for ls

    lmbdm = np.zeros((noutliers, nlmbd))  # every proportion of outliers need a different lambda
    lmbdm[0, :] = np.logspace(-1, 1, nlmbd)  # lambdas for ls
    lmbdm[1, :] = np.logspace(-1, 2, nlmbd)  # lambdas for ls
    lmbdm[2, :] = np.logspace(-1, 2, nlmbd)  # lambdas for ls
    lmbdm[3, :] = np.logspace(1, 3.5, nlmbd)  # lambdas for ls
    lmbdm[4, :] = np.logspace(1, 4, nlmbd)  # lambdas for ls

    lmbdmes = np.zeros((noutliers, nlmbd))  # every proportion of outliers need a different lambda
    lmbdmes[0, :] = np.logspace(1, 4, nlmbd)  # lambdas for ls
    lmbdmes[1, :] = np.logspace(4, 6, nlmbd)  # lambdas for ls
    lmbdmes[2, :] = np.logspace(4, 6, nlmbd)  # lambdas for ls
    lmbdmes[3, :] = np.logspace(4, 6, nlmbd)  # lambdas for ls
    lmbdmes[4, :] = np.logspace(4, 6, nlmbd)  # lambdas for ls

    lmbdtau = np.zeros((noutliers, nlmbd))  # every proportion of outliers need a different lambda
    lmbdtau[0, :] = np.logspace(-2, 1, nlmbd)  # lambdas for ls
    lmbdtau[1, :] = np.logspace(-2, 2, nlmbd)  # lambdas for ls
    lmbdtau[2, :] = np.logspace(-1, 2, nlmbd)  # lambdas for ls
    lmbdtau[3, :] = np.logspace(0, 2, nlmbd)  # lambdas for ls
    lmbdtau[4, :] = np.logspace(2, 4, nlmbd)  # lambdas for ls

    errorls = np.zeros((noutliers, nlmbd, nrealizations))  # store results for ls
    errormes = np.zeros((noutliers, nlmbd, nrealizations))  # store results for m with an estimated scale
    errorm = np.zeros((noutliers, nlmbd, nrealizations))  # store results for m
    errortau = np.zeros((noutliers, nlmbd, nrealizations))  # store results for tau
    k = 0
    while k < noutliers:
        t = 0
        print 'outliers % s' % k
        while t < nlmbd:
            print 'lambda % s' % t
            r = 0
            while r < nrealizations:
                y = util.getmeasurements(a, x, noisetype, var, outliers[k])  # get the measurements
                # -------- ls solution
                xhat = inv.ridge(y, a, lmbdls[k, t])  # solving the problem with ls
                error = np.linalg.norm(x - xhat)
                errorls[k, t, r] = error
                # -------- m estimated scale solution
                xpreliminary = xhat  # we take the ls to estimate a preliminary scale
                respreliminary = y - np.dot(a, xpreliminary)
                estimatedscale = np.median(np.abs(respreliminary)) / .6745  # robust mad estimator for the scale
                xhat = inv.mridge(y, a, 'huber', clippinghuber, estimatedscale, lmbdmes[k, t])  # solving the problem with m
                error = np.linalg.norm(x - xhat)
                errormes[k, t, r] = error
                # -------- m real scale solution
                xhat = inv.mridge(y, a, 'huber', clippinghuber, realscale, lmbdm[k, t])  # solving the problem with m
                error = np.linalg.norm(x - xhat)
                errorm[k, t, r] = error
                # -------- tau solution

                xhat, scale = ln.basictau(
                    a,
                    y,
                    'optimal',
                    clippingopt,
                    ninitialx=ninitialsolutions,
                    maxiter=100,
                    nbest=1,
                    regularization=ln.tikhonov_regularization,
                    lamb=lmbdtau[k, t]
                )

                error = np.linalg.norm(x - xhat)
                errortau[k, t, r] = error
                r += 1  # update the number of realization
            t += 1  # update the number of lambda that we are trying
        k += 1  # updated the number of outlier proportion

    minls = np.min(errorls, 1)
    minm = np.min(errorm, 1)
    minmes = np.min(errormes, 1)
    mintau = np.min(errortau, 1)

    avgls = np.mean(minls, 1)
    avgm = np.mean(minm, 1)
    avgmes = np.mean(minmes, 1)
    avgtau = np.mean(mintau, 1)

    fone, axone = plt.subplots(noutliers, sharex=True)  # plots to check if we are getting the best lambda
    cnt = 0
    while cnt < noutliers:
      axone[cnt].plot(lmbdls[0, :], errorls[cnt, :, 1])
      axone[cnt].set_xscale('log')
      cnt += 1
    axone[0].set_title('LS')
    # plt.show()

    ftwo, axtwo = plt.subplots(noutliers, sharex=True)  # plots to check if we are getting the best lambda
    cnt = 0
    while cnt < noutliers:
      axtwo[cnt].plot(lmbdls[0, :], errorm[cnt, :, 1])
      axtwo[cnt].set_xscale('log')
      cnt += 1
    axtwo[0].set_title('M estimator')
    # plt.show()

    fthree, axthree = plt.subplots(noutliers, sharex=True)  # plots to check if we are getting the best lambda
    cnt = 0
    while cnt < noutliers:
      axthree[cnt].plot(lmbdmes[0, :], errormes[cnt, :, 1])
      axthree[cnt].set_xscale('log')
      cnt += 1
    axthree[0].set_title('M estimator est. scale')
    # plt.show()

    ffour, axfour = plt.subplots(noutliers, sharex=True)  # plots to check if we are getting the best lambda
    cnt = 0
    while cnt < noutliers:
      axfour[cnt].plot(lmbdtau[0, :], errortau[cnt, :, 1])
      axfour[cnt].set_xscale('log')
      cnt += 1
    axfour[0].set_title('tau estimator')
    # plt.show()

    # store results
    name_file = 'experiment_two.pkl'
    fl = os.path.join(DATA_DIR, name_file)
    f = open(fl, 'wb')
    pickle.dump([avgls, avgm, avgmes, avgtau], f)
    f.close()

    fig = plt.figure()
    plt.plot(outliers, avgls, 'r--', label='ls')
    plt.plot(outliers, avgm, 'bs--', label='m estimator')
    plt.plot(outliers, avgmes, 'g^-', label='m est. scale')
    plt.plot(outliers, avgtau, 'kd-', label='tau')
    plt.legend(loc=2)
    plt.xlabel('% outliers')
    plt.ylabel('error')

    name_file = 'experiment_two.eps'
    fl = os.path.join(FIGURES_DIR, name_file)
    fig.savefig(fl, format='eps')

    # plt.show()  # show figure


# -------------------------------------------------------------------
# Experiment 3: l1 regularized case. Comparison LS, M, tau
# -------------------------------------------------------------------
def experimentthree(nrealizations, outliers, measurementsize, sourcesize, source):

    sourcetype = 'sparse'  # kind of source we want
    sparsity = 0.2
    matrixtype = 'illposed'  # type of sensing matrix
    conditionnumber = 1000  # condition number of the matrix that we want
    noisetype = 'outliers'  # additive noise
    var = 3  # variance of the noise
    clippinghuber = 1.345  # clipping parameter for the huber function
    clippingopt = (0.4, 1.09)  # clipping parameters for the opt function in the tau estimator
    ninitialsolutions = 10  # how many initial solutions do we want in the tau estimator
    maxiter = 50
    nlmbd = 5  # how many different lambdas are we trying in each case
    realscale = 1
    x = source

    print '||x|| = ', np.linalg.norm(x)
    a = util.getmatrix(sourcesize, matrixtype, measurementsize, conditionnumber)  # get the sensing matrix
    noutliers = outliers.size
    scaling = 1e-2
    realscale *= scaling


    if scaling == 1e-2:
        lmbdls = np.zeros((noutliers, nlmbd))  # every proportion of outliers need a different lambda
        lmbdls[0, :] = np.logspace(-2, 3, nlmbd)  # lambdas for ls
        lmbdls[1, :] = np.logspace(3, 5, nlmbd)  # lambdas for ls
        lmbdls[2, :] = np.logspace(3.5, 5, nlmbd)  # lambdas for ls
        lmbdls[3, :] = np.logspace(4, 5, nlmbd)  # lambdas for ls
        lmbdls[4, :] = np.logspace(4, 5, nlmbd)  # lambdas for ls

        lmbdlm = np.zeros((noutliers, nlmbd))  # every proportion of outliers need a different lambda
        lmbdlm[0, :] = np.logspace(-3, 3, nlmbd)  # lambdas for m
        lmbdlm[1, :] = np.logspace(-3, 3.5, nlmbd)  # lambdas for m
        lmbdlm[2, :] = np.logspace(-3, 4, nlmbd)  # lambdas for m
        lmbdlm[3, :] = np.logspace(-3, 4, nlmbd)  # lambdas for m
        lmbdlm[4, :] = np.logspace(-3, 4, nlmbd)  # lambdas for m

        lmbdlmes = np.zeros((noutliers, nlmbd))  # every proportion of outliers need a different lambda
        lmbdlmes[0, :] = np.logspace(-3, 3, nlmbd)  # lambdas for m with est. scale
        lmbdlmes[1, :] = np.logspace(2, 3.5, nlmbd)  # lambdas for m with est. scale
        lmbdlmes[2, :] = np.logspace(2.2, 3.5, nlmbd)  # lambdas for m with est. scale
        lmbdlmes[3, :] = np.logspace(3, 4, nlmbd)  # lambdas for m with est. scale
        lmbdlmes[4, :] = np.logspace(3.5, 4.5, nlmbd)  # lambdas for m with est. scale

        lmbdltau = np.zeros((noutliers, nlmbd))  # every proportion of outliers need a different lambda
        lmbdltau[0, :] = np.logspace(-3.5, 1, nlmbd)  # lambdas for tau
        lmbdltau[1, :] = np.logspace(-3, 1, nlmbd)  # lambdas for tau
        lmbdltau[2, :] = np.logspace(-3.5, 1, nlmbd)  # lambdas for tau
        lmbdltau[3, :] = np.logspace(-3.5, 1, nlmbd)  # lambdas for tau
        lmbdltau[4, :] = np.logspace(-1.5, 2, nlmbd)  # lambdas for tau

    else:
        lmbdls = np.zeros((noutliers, nlmbd))  # every proportion of outliers need a different lambda
        lmbdls[0, :] = np.logspace(2, 6, nlmbd)  # lambdas for ls
        lmbdls[1, :] = np.logspace(7, 9, nlmbd)  # lambdas for ls
        lmbdls[2, :] = np.logspace(7, 9, nlmbd)  # lambdas for ls
        lmbdls[3, :] = np.logspace(7, 9, nlmbd)  # lambdas for ls
        lmbdls[4, :] = np.logspace(6, 9, nlmbd)  # lambdas for ls

        lmbdlm = np.zeros((noutliers, nlmbd))  # every proportion of outliers need a different lambda
        lmbdlm[0, :] = np.logspace(-2, 3.5, nlmbd)  # lambdas for m
        lmbdlm[1, :] = np.logspace(-2, 3.5, nlmbd)  # lambdas for m
        lmbdlm[2, :] = np.logspace(-2, 4, nlmbd)  # lambdas for m
        lmbdlm[3, :] = np.logspace(-2, 4, nlmbd)  # lambdas for m
        lmbdlm[4, :] = np.logspace(-2, 4, nlmbd)  # lambdas for m

        lmbdlmes = np.zeros((noutliers, nlmbd))  # every proportion of outliers need a different lambda
        lmbdlmes[0, :] = np.logspace(-1, 4, nlmbd)  # lambdas for m with est. scale
        lmbdlmes[1, :] = np.logspace(-0.5, 4, nlmbd)  # lambdas for m with est. scale
        lmbdlmes[2, :] = np.logspace(-1, 5, nlmbd)  # lambdas for m with est. scale
        lmbdlmes[3, :] = np.logspace(-1, 5, nlmbd)  # lambdas for m with est. scale
        lmbdlmes[4, :] = np.logspace(0, 6, nlmbd)  # lambdas for m with est. scale

        lmbdltau = np.zeros((noutliers, nlmbd))  # every proportion of outliers need a different lambda
        lmbdltau[0, :] = np.logspace(-2, 2, nlmbd)  # lambdas for tau
        lmbdltau[1, :] = np.logspace(-2, 2, nlmbd)  # lambdas for tau
        lmbdltau[2, :] = np.logspace(-1.5, 2, nlmbd)  # lambdas for tau
        lmbdltau[3, :] = np.logspace(-1.5, 2.5, nlmbd)  # lambdas for tau
        lmbdltau[4, :] = np.logspace(-1.5, 3, nlmbd)  # lambdas for tau

    errorls = np.zeros((noutliers, nlmbd, nrealizations))  # store results for ls
    errormes = np.zeros((noutliers, nlmbd, nrealizations))  # store results for m with an estimated scale
    errorm = np.zeros((noutliers, nlmbd, nrealizations))  # store results for m
    errortau = np.zeros((noutliers, nlmbd, nrealizations))  # store results for tau

    k = 0
    while k < noutliers:
        print 'number of outliers %s' % k
        t = 0
        while t < nlmbd:
            print 'lambdas %s' % t
            r = 0
            while r < nrealizations:
                y = util.getmeasurements(a, x, noisetype, var, outliers[k])  # get the measurements
                ascaled = a * scaling  # scaling the data to avoid numerical problems with cvx
                yscaled = y * scaling

                #  -------- ls solution
                #xhat = inv.lasso(yscaled, ascaled, lmbdls[k, t])  # solving the problem with ls
                xhat = ln.lasso_regularization(ascaled, yscaled, lambda_parameter=lmbdls[k, t])  # solving the problem with ls
                xhat = xhat.reshape(-1, 1)
                error = np.linalg.norm((x - xhat))
                errorls[k, t, r] = error

                # -------- m estimated scale solution
                xpreliminary = xhat  # we take the ls to estimate a preliminary scale
                # respreliminary = y - np.dot(a, xpreliminary)
                respreliminary = yscaled - np.dot(ascaled, xpreliminary)
                estimatedscale = np.median(np.abs(respreliminary)) / .6745  # robust mad estimator for the scale
                # xhat = inv.mlasso(yscaled, ascaled, 'huber', clippinghuber, estimatedscale, lmbdlmes[k, t])  # solving the problem with m
                xhat = ln.m_estimator(
                    ascaled,
                    yscaled,
                    'optimal',
                    clippinghuber,
                    estimatedscale,
                    regularization=ln.lasso_regularization,
                    lmbd=lmbdlmes[k, t]
                )
                xhat = xhat.reshape(-1, 1)
                error = np.linalg.norm(x - xhat)
                errormes[k, t, r] = error


                #  -------- m real scale solution
                # xhat = inv.mlasso(yscaled, ascaled, 'huber', clippinghuber, realscale, lmbdlm[k, t])  # solving the problem with m
                xhat = ln.m_estimator(
                    ascaled,
                    yscaled,
                    'optimal',
                    clippinghuber,
                    realscale,
                    regularization=ln.lasso_regularization,
                    lmbd=lmbdlm[k, t]
                )
                xhat = xhat.reshape(-1, 1)
                error = np.linalg.norm(x - xhat)
                errorm[k, t, r] = error

                #  -------- tau solution
                # xhat, scale = inv.fasttaulasso(yscaled, ascaled, 'optimal', clippingopt, ninitialsolutions, lmbdltau[k, t], maxiter)
                xhat, scale = ln.basictau(
                    ascaled,
                    yscaled,
                    'optimal',
                    clippingopt,
                    ninitialx=ninitialsolutions,
                    maxiter=maxiter,
                    nbest=1,
                    regularization=ln.lasso_regularization,
                    lamb=lmbdltau[k, t]
                )
                xhat = xhat.reshape(-1, 1)
                error = np.linalg.norm(x - xhat)
                errortau[k, t, r] = error

                print 'error = ', error
                print 'error shape =', (x - xhat).shape
                print '% of outliers =', outliers[k]
                print 'lambda idx = ', t
                print '---------------------'

                r += 1  # update the number of realization]
            t += 1  # update the number of realization
        k += 1  # updated the number of outlier proportion

    minls = np.min(errorls, 1)
    minm = np.min(errorm, 1)
    minmes = np.min(errormes, 1)
    mintau = np.min(errortau, 1)

    avgls = np.mean(minls, 1)
    avgm = np.mean(minm, 1)
    avgmes = np.mean(minmes, 1)
    avgtau = np.mean(mintau, 1)

    pickle.dump(avgls, open("ls.p", "wb"))
    pickle.dump(avgm, open("m.p", "wb"))
    pickle.dump(avgmes, open("mes.p", "wb"))
    pickle.dump(avgtau, open("tau.p", "wb"))

    fthree, axthree = plt.subplots(noutliers, sharex=True)  # plots to check if we are getting the best lambda
    cnt = 0
    while cnt < noutliers:
        axthree[cnt].plot(lmbdlmes[0, :], errormes[cnt, :, 1])
        axthree[cnt].set_xscale('log')
        cnt += 1
    axthree[0].set_title('M estimator estimated scale')
    plt.show()

    # fthree, axthree = plt.subplots(noutliers, sharex=True)  # plots to check if we are getting the best lambda
    # cnt = 0
    # while cnt < noutliers:
    #   axthree[cnt].plot(lmbdlmes[0, :], errortau[cnt, :, 0])
    #   axthree[cnt].set_xscale('log')
    #   cnt += 1
    # axthree[0].set_title('M estimator est.0')
    # plt.show()
    #
    # fthree, axthree = plt.subplots(noutliers, sharex=True)  # plots to check if we are getting the best lambda
    # cnt = 0
    # while cnt < noutliers:
    #   axthree[cnt].plot(lmbdlmes[0, :], errorm[cnt, :, 1])
    #   axthree[cnt].set_xscale('log')
    #   cnt += 1
    # axthree[0].set_title('Mmm estimator est 1')
    # plt.show()
    #
    # fthree, axthree = plt.subplots(noutliers, sharex=True)  # plots to check if we are getting the best lambda
    # cnt = 0
    # while cnt < noutliers:
    #   axthree[cnt].plot(lmbdlmes[0, :], errorm[cnt, :, 0])
    #   axthree[cnt].set_xscale('log')
    #   cnt += 1
    # axthree[0].set_title('Mmm estimator est.0')
    #plt.show()

    # fthree, axthree = plt.subplots(noutliers, sharex=True)  # plots to check if we are getting the best lambda
    # cnt = 0
    # while cnt < noutliers:
    #   axthree[cnt].plot(lmbdlmes[0, :], errortau[cnt, :, 2])
    #   axthree[cnt].set_xscale('log')
    #   cnt += 1
    # axthree[0].set_title('M estimator est. 1')
    # plt.show()



    # ffour, axfour = plt.subplots(noutliers, sharex=True)  # plots to check if we are getting the best lambda
    # cnt = 0
    # while cnt < noutliers:
    #   axfour[cnt].plot(lmbdltau[0, :], errortau[cnt, :, 1])
    #   axfour[cnt].set_xscale('log')
    #   cnt += 1
    # axfour[0].set_title('tau estimator')
    # plt.show()

    # store results
    name_file = 'experiment_three.pkl'
    fl = os.path.join(DATA_DIR, name_file)
    f = open(fl, 'wb')
    pickle.dump([avgls, avgm, avgmes, avgtau], f)
    f.close()

    fig = plt.figure()
    plt.plot(outliers, avgls, 'r--', label='ls')
    plt.plot(outliers, avgm, 'bs--', label='m estimator')
    plt.plot(outliers, avgmes, 'g^-', label='m est. scale')
    plt.plot(outliers, avgtau, 'kd-', label='tau')
    plt.legend(loc=2)  # plot legend
    plt.xlabel('% outliers')  # plot x label
    plt.ylabel('error')  # plot y label

    name_file = 'experiment_three.eps'
    fl = os.path.join(FIGURES_DIR, name_file)
    fig.savefig(fl, format='eps')

    # plt.show()  # show figure


def comparison_irls_apg(nruns, proximal_operator, reg):
    """
    In this function we compare the results of apg and irls algorithms for l2-norm regularization.
    Out of nruns of the experiment, the output is the maximum relative distance (||x_apg - x_irls|| / ||x_groundtruth||)
    :param nruns: number of runs of the experiment
    :param proximal_operator: function that computes prox operator
    :param reg: function that gives ls + reg result
    :return: maximum relative error
    """

    m = 10  # number of measurements
    n = 3  # dimensions of x
    xs_apg = np.zeros((nruns, n))
    xs_irls = np.zeros((nruns, n))

    for iteration in range(nruns):
        # defining linear problem, create data
        a = np.random.rand(m, n)  # model matrix
        x_grountruth = 10 * np.squeeze(np.random.rand(n, 1))  # ground truth
        x_grountruth = x_grountruth.tolist()
        y = np.dot(a, x_grountruth).reshape(-1, 1) + 0.5 * np.random.rand(m, 1)
        print x_grountruth

        # define any x
        x = np.squeeze(10 * np.random.rand(n, 1))
        x = x.tolist()

        # clipping parameters
        clipping_1 = 1.21
        clipping_2 = 3.27

        # regularization parameter
        reg_parameter = 0.00001

        # run apg algorithms
        x_apg, objDistance, alpha_l, alpha_x = ln.tau_apg(
            a,
            y,
            reg_parameter,
            clipping_1,
            clipping_2,
            x,
            proximal_operator,
            rtol=1e-10
        )

        # run irls algorithm
        initx = np.array(x)
        initx = initx.reshape(-1, 1)
        x_irls = ln.basictau(
            a,
            y,
            'optimal',
            [clipping_1, clipping_2],
            ninitialx=0,
            maxiter=200,
            nbest=1,
            initialx=initx,
            b=0.5,
            regularization=reg,
            lamb=reg_parameter
        )

        # store in the general array
        xs_apg[iteration, :] = x_apg
        print 'x_apg = ', x_apg
        print 'obj step = ', objDistance
        print 'alphas = ', [alpha_l, alpha_x]
        print 'x_irls = ', x_irls[0]
        print '==================='

        xs_irls[iteration, :] = np.squeeze(x_irls[0])

    # norm of the error
    distance = np.linalg.norm(xs_irls - xs_apg, axis=1)

    # maximum distance
    max_distance = np.max(distance)

    # relative distance
    rdistance = max_distance / np.linalg.norm(x_grountruth)

    return rdistance, max_distance


def error_if_sc():
    mat = scipy.io.loadmat('./mathematica_data/IFtauNonReg.mat')
    IF = mat['Expression1']

    name_file = 'sc.p'
    fl = os.path.join(DATA_DIR, name_file)
    f = open(fl, 'rb')
    sc = pickle.load(f)

    print 'sc shape', sc.shape
    print 'if shape', IF.shape


def main(argv):

    def do_sc():
        #points = 20
        points = 21
        yrange = 10
        arange = 10
        nmeasurements = 1000
        sensitivitycurve('tau', 'optimal', 'none', yrange, arange, nmeasurements, points)

    def do_asv():
        lrange = 1
        lstep = 0.1
        nrealizations = 100
        asv('tau', 'l2', lrange, lstep, nrealizations)
        asv('tau', 'l1', lrange, lstep, nrealizations)

    def do_bias():
        lrange = 0.50
        lstep = 0.05
        nrealizations = 100
        bias('tau', 'l2', lrange, lstep, nrealizations)
        bias('tau', 'l1', lrange, lstep, nrealizations)

    def do_experiment_one():
        outliers = np.linspace(0, 0.4, 5)
        measurementsize = 60
        sourcesize = 20
        numberofruns = 50
        x = pickle.load(open("sourcetwo.p", "rb"))  # getting x from file
        experimentone(numberofruns, outliers, measurementsize, sourcesize, x)

    def do_experiment_two():
        outliers = np.linspace(0, 0.4, 5)
        measurementsize = 60
        sourcesize = 20
        numberofruns = 50
        x = pickle.load(open("sourcetwo.p", "rb"))  # getting x from file
        experimenttwo(numberofruns, outliers, measurementsize, sourcesize, x)

    def do_experiment_three():
        outliers = np.linspace(0, 0.4, 5)
        measurementsize = 60
        sourcesize = 20
        numberofruns = 100
        x = pickle.load(open("sourcethree.p", "rb"))  # getting x from file
        experimentthree(numberofruns, outliers, measurementsize, sourcesize, x)

    def do_test():
        error_if_sc()

    def do_real_data(list_estimators):
        # load data
        data = scipy.io.loadmat('./matlab_data/experimentalData.mat')

        # fetch data
        a = data['M']  # model matrix
        y = data['y']  # measurements
        x = data['x']  # ground truth for the source

        # scaling data to avoid numerical problems. Remaned to not forget the scaling
        scaling_constant = 1e15
        a_scaled = scaling_constant * a
        y_scaled = scaling_constant * y

        # define range of exploration for lambda
        nlmbd = 20  # how many lambdas do we want to test
        #start = 0  # starting point
        #end = 4  # end point
        start = -7
        end = -5
        reg_parameter = np.logspace(start, end, nlmbd)


        # store the estimates
        xs_l2 = np.zeros((nlmbd,  x.shape[0]))
        xs_l1 = np.zeros((nlmbd,  x.shape[0]))
        xs_ls = np.zeros((nlmbd,  x.shape[0]))

        # parameters for the tau estimator
        # clipping_1 = 0.4
        # clipping_2 = 1.09

        start_c1 = 1.5
        end_c1 = 2  # 5
        start_c2 = 3
        end_c2 = 4  # 20
        n_cs = 10
        clipping_1 = np.linspace(start_c1, end_c1, n_cs)
        clipping_2 = np.linspace(start_c2, end_c2, n_cs)

        # init. vector to store results
        error_l2 = np.zeros((nlmbd, n_cs, n_cs))
        error_l1 = np.zeros((nlmbd, 1))
        error_ls = np.zeros((nlmbd, 1))

        initial_solution = np.ones(x.shape)

        for idx, lb in enumerate(reg_parameter):
            print '============= lambda idx =', idx

            if 'tikhonov' in list_estimators:
                # solve the problem using just Tikhonov
                x_ls = ln.tikhonov_regularization(a_scaled, y_scaled, lambda_parameter=lb)
                x_ls = x_ls.reshape(-1, 1)

                # compute the error and store results
                error_ls[idx] = np.linalg.norm(x_ls - x)

                # save x
                xs_ls[idx, :] = np.squeeze(x_ls)

            elif 'tau_l2' in list_estimators:
                for idx_c1, c1 in enumerate(clipping_1):
                    # look for the best clipping1 parameter
                    print '============= c1 idx =', idx_c1
                    for idx_c2, c2 in enumerate(clipping_2):
                        # look for the best clipping2 parameter
                        name_file = 'initial_x_ls.pkl'
                        fl = os.path.join(DATA_DIR, name_file)
                        f = open(fl, 'rb')
                        initial_solution = pickle.load(f)  # getting x from file
                        initial_solution = initial_solution.reshape(-1, 1)

                        # solve the problem with tau and two different regularizations
                        x_irls_l2, scale = ln.basictau(
                            a_scaled,
                            y_scaled,
                            'optimal',
                            [c1, c2],
                            ninitialx=0,
                            initialx=initial_solution,
                            maxiter=200,
                            nbest=1,
                            b=0.5,
                            regularization=ln.tikhonov_regularization,
                            lamb=lb
                        )

                        error_l2[idx, idx_c1, idx_c2] = np.linalg.norm(x_irls_l2 - x)
            else:
                x_irls_l1, scale = ln.basictau(
                    a_scaled,
                    y_scaled,
                    'optimal',
                    [clipping_1, clipping_2],
                    ninitialx=0,
                    initialx=initial_solution,
                    maxiter=200,
                    nbest=1,
                    b=0.5,
                    regularization=ln.lasso_regularization,
                    lamb=lb
                )
                error_l1[idx] = np.linalg.norm(x_irls_l1 - x)

        # store results
        name_file = 'errors_tau_l2.pkl'
        fl = os.path.join(DATA_DIR, name_file)
        f = open(fl, 'wb')
        pickle.dump([error_l2], f)
        f.close()

        # find minimum value
        opt_ls = np.min(error_ls)
        opt_tau_l2 = np.min(error_l2)
        opt_tau_l1 = np.min(error_l1)

        # save best x
        # best_x_idx = np.argmin(error_ls)
        # best_x = xs_ls[best_x_idx, :]

        # store results
        # name_file = 'initial_x_ls.pkl'
        # fl = os.path.join(DATA_DIR, name_file)
        # f = open(fl, 'wb')
        # pickle.dump(best_x, f)
        # f.close()

        # print in terminal
        print "LS error = ", opt_ls
        print "tau-l2 error = ", opt_tau_l2
        print "tau-l1 error = ", opt_tau_l1

        # make plot
        # fig = plt.figure()
        # plt.semilogx(reg_parameter, error_ls, '-o', color='black', label='tik')
        # plt.semilogx(reg_parameter, error_l2, '-o', color='blue', label='tau-l2')
        # plt.semilogx(reg_parameter, error_l1, '-o', color='red', label='tau-l1')
        # plt.legend()
        # plt.xlabel('regularization parameter')
        # plt.ylabel('||x -xhat||')
        #
        # name_file = 'real_data_regression.eps'
        # fl = os.path.join(FIGURES_DIR, name_file)
        # fig.savefig(fl, format='eps')

    def do_fine_search():
        """
        Do a fine search for the parameters with the real dataset

        :param list_estimators:
        :return:
        """
        # open results
        name_file = 'errors_tau_l2.pkl'
        fl = os.path.join(DATA_DIR, name_file)
        f = open(fl, 'rb')
        errors_tau_ls = np.squeeze(np.array(pickle.load(f)))  # getting x from file
        indx_min = np.argmin(errors_tau_ls)
        indx_abs = np.unravel_index(indx_min, errors_tau_ls.shape)
        print 'index for the min value = ', indx_abs

        start_c1 = 1.8
        end_c1 = 1.9  # 5
        start_c2 = 3.5
        end_c2 = 3.6  # 20
        n_cs = 10
        clipping_1 = np.linspace(start_c1, end_c1, n_cs)
        clipping_2 = np.linspace(start_c2, end_c2, n_cs)

        nlmbd = 20  # how many lambdas do we want to test
        start = -7
        end = -6
        reg_parameter = np.logspace(start, end, nlmbd)

        print 'minimum error = ', np.min(errors_tau_ls)
        print 'clipping_1 optimal value = ', clipping_1[indx_abs[1]]
        print 'clipping_2 optimal value = ', clipping_2[indx_abs[2]]
        print 'reg_parameter optimal value = ', reg_parameter[indx_abs[0]]

    def do_convert_to_matlab():
        # open results
        name_file = 'sc_l1.pkl'
        fl = os.path.join(DATA_DIR, name_file)
        f = open(fl, 'rb')
        sc = pickle.load(f)

        # convert them to a matlab file and save it
        mat_fl = os.path.join(DATA_DIR, 'sc.mat')
        scipy.io.savemat(mat_fl, dict(sc=sc))




    # parse inline arguments
    FLAGS(argv)

    dict(
        bias=lambda: do_bias(),
        asv=lambda: do_asv(),
        sc=lambda: do_sc(),
        experiment_one=lambda: do_experiment_one(),
        experiment_two=lambda: do_experiment_two(),
        experiment_three=lambda: do_experiment_three(),
        test=lambda: do_test(),
        real_data=lambda: do_real_data(['tau_l2']),
        search_min=lambda: do_fine_search(),
        convert_matlab=lambda: do_convert_to_matlab()
    )[FLAGS.figure]()


if __name__ == '__main__':
    # To launch manually in console:
    # python toolboxexperiments.py --figure='your_figure'
    main(sys.argv)




