import linvpy as lp
import mestimator_marta as marta
import generate_random as gen
import numpy as np
import matplotlib.pyplot as plt
import optimal as opt
from scipy.sparse.linalg import lsmr

A = np.matrix([[1,3],[3,4]])
y = np.array([-6,1,-2])

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
'''


A,y = gen.generate_random(4)
#print "my gradient= ", lp.least_squares_gradient(A,y)
#print "my matrice form= ", lp.least_squares(A,y) 
#print "numpy's =", np.linalg.lstsq(A,y)[0]

LAMBDA=0.2
print lp.tikhonov_regularization(A,y,LAMBDA)
print lsmr(A,y,LAMBDA)[0]





