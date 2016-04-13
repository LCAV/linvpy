import linvpy as lp
import mestimator_marta as marta
import generate_random as gen
import numpy as np
import matplotlib.pyplot as plt
import optimal as opt
from scipy.sparse.linalg import lsmr
import toolboxutilities
import toolboxinverse

A,y = gen.generate_random(4)

#x = np.array[1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0]

#x = np.array(1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0)

#x = np.array[1.0 2.0 3.0 4.0 5.0 6.0 7.0 8.0 9.0]

'''
print [reg.psi_huber(e,5) for e in x]

print "marta huber =", marta.huber(x,3)

A,y = gen.generate_random(3)

B = np.matrix([[1,2,0],[1,2,0],[1,2,3]])


# print "Marta's = ", marta.irls(y,A,'huber',[1, 1],1.5)

print "my psi_huber =", [reg.psi_huber(e,3) for e in x]
print "x=", x
print "my weights =", reg.weights(x, reg.psi_huber, 3)

print marta.weights(B, 'M', 'huber', 1.5, 0)

print reg.rho_bisquare(2,5)
print [reg.rho_bisquare(i,5) for i in x]


plt.plot([reg.psi_cauchy(i) for i in range(-20,20)])
plt.show()

print [reg.psi_bisquare(i,5) for i in x]

marta_list = opt.rhooptimal(y,2)
my_list = [lp.rho_optimal(i) for i in y]

marta_numpy = np.array(marta_list)
my_numpy = np.array(my_list)

print "marta's rho : ", opt.rhooptimal(y,2)
print "marta's score : ", opt.scoreoptimal(y,2)
#print reg.rho_optimal(2,3)

print "mine : ", [lp.rho_optimal(i) for i in y]


A,y = gen.generate_random(4)
#print "my gradient= ", lp.least_squares_gradient(A,y)
#print "my matrice form= ", lp.least_squares(A,y) 
#print "numpy's =", np.linalg.lstsq(A,y)[0]

LAMBDA=0.2
print lp.tikhonov_regularization(A,y,LAMBDA)
print lsmr(A,y,LAMBDA)[0]


x = [1,2,3,4,5,6,7,8,9]

result = [lp.psi_cauchy(e, 4) for e in x]

# [0.9411764705882353, 1.6, 1.92, 2.0, 1.951219512195122, 1.8461538461538463, 1.7230769230769232, 1.6, 1.4845360824742269]

print result

def lol(a,b):
	return a+b

def func(function,a,b):
	return function(a,b)


x = [1,2,3,4,0]

#print lp.weights(x, lp.rho_cauchy, -2)
print [0 if (i == 0) else lp.rho_huber(i,2)/float(i) for i in x]

print lp.rho_huber(2)/float(2)
print lp.weights(2, lp.rho_huber)
#print lp.weights(1, lp.psi_huber)
#print lp.psi_huber(2)



print toolboxutilities.huber(x,1.5)
print [lp.psi_huber(i,1.5) for i in x]


print toolboxinverse.ridge(y,A,0.5)
print lp.tikhonov_regularization(A,y,0.5)

A,y = gen.generate_random(5)
B,x =gen.generate_random(4)

'''
print np.ones(4)
print "y= ", y
print "A = ", A
print "dot result = ", np.dot(A, np.ones(3))

print lp.irls(A,y, lp.psi_huber)

print toolboxinverse.mestimator(y,A, 'huber', 3, np.ones(3))

#print toolboxinverse.irls(y,A,'M', 'huber', 'none', 0.5, np.ones(3), 0,1.5)

#def irls(y, a, kind, lossfunction, regularization, lmbd, initialx, initialscale, clipping, maxiter=100, tolerance=1e-5,


'''
Iterative Re-weighted Least Squares algorithm

Input arguments:
y: measurements
a: model matrix
kind: type of method (M)
lossfunction: type of loss function that we want (squared, huber)
regularization: type of regularization. Options: none, l2
initialx: initial solution
initialscale: initial scale. For the M estimator the scale is fixed, so in this case this is the preliminary scale
clipping: clipping parameter for the loss function
lmbd: regularization parameter
tolerance: if two consecutive iterations give two solutions closer than tolerance, the algorithm stops
maxiter: maximum number of iterations. If the algorithm reaches this, it stops

#irls(y, a, 'M', lossfunction, 'none', 0, initialx, preliminaryscale, clipping)

print opt.rhooptimal(y,4.5)

print [lp.rho_optimal(i,4.5) for i in y]
LAMBDA = 2
print np.asarray([lp.rho_optimal(i,LAMBDA) for i in y]), opt.rhooptimal(y,LAMBDA)


plt.plot([lp.rho_optimal(i, 3.27) for i in range(-10,10)], label="rho_huber")
plt.plot([lp.rho_optimal(i, 1.21) for i in range(-10,10)], label="rho_huber")
plt.show()

x = [1,2,3,4,5,6,7,8,9]
y = np.array(x)

print lp.weights(x, lp.psi_huber,3)
print toolboxutilities.weights(y, 'M', 'squared', 3, 0)
'''



