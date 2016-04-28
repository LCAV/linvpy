import linvpy as lp
import mestimator_marta as marta
import generate_random as gen
import numpy as np
import matplotlib.pyplot as plt
import optimal as opt
from scipy.sparse.linalg import lsmr
import toolboxutilities
import toolboxinverse



#genA, geny = gen.generate_random(4,5)
#print "test =", lp.irls(genA, geny, lp.psi_huber)

#28.04.16

c = np.ones((2, 1))
d = np.ones((2, 2))

print "C = ", c
print "D = ", d


y_gui = x = np.array([3,5])
A_gui = np.array([[1, 2], [3, 4]])



new_y = np.array([[2],[3]])
new_a = np.array([[3,3],[2,3]])
m, n = new_a.shape
initialx = np.ones((n, 1))  # initial solution

print "Marta's M-estimator = ", marta.mestimator(new_y, new_a, 'huber',0.5)[0]

print "MARTA's IRLS = ", marta.irls(new_y, new_a, 'huber', initialx, 0.5)[0]


last_y = [2,3]
last_a = np.array([[3,3],[2,3]])

print "My m-estimator = ", lp.irls(last_a, last_y, lp.psi_huber, clipping=1.2,
	lamb=0, scale=300)





# def mestimator(y, a, lossfunction, clipping):

# for a large clipping parameter, LS and M outputs should be the same



'''
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


'''

'''
A,y = gen.generate_random(5,4)



# TEST WEIGHTS => SAME BUT DIVIDED BY 2

print "my weights =", lp.weights(y, lp.psi_huber)

print "marta's weights = ", marta.weights(y, 'M', 'huber', 0.1, 0)
'''

